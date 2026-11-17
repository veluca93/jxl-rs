// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use proc_macro::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};

use crate::{
    GenericKind, GenericParam, ParsedSimdFn, bounded_ty_starts_with, err, find_single_colon,
    flatten_none, is_punct, parse_simd_fn, path_tokens, span_of, split_top_level_commas, ts,
};

#[derive(Clone, Copy)]
#[repr(u8)]
enum Isa {
    Emu128 = 0,
    Sse42 = 1,
    Avx = 2,
    Avx512 = 3,
    Neon = 4,
    Simd128 = 5,
}

impl Isa {
    fn from_ident(id: &Ident) -> Result<Self, TokenStream> {
        match id.to_string().as_str() {
            "Emu128Descriptor" => Ok(Self::Emu128),
            "Sse42Descriptor" => Ok(Self::Sse42),
            "AvxDescriptor" => Ok(Self::Avx),
            "Avx512Descriptor" => Ok(Self::Avx512),
            "NeonDescriptor" => Ok(Self::Neon),
            "Simd128Descriptor" => Ok(Self::Simd128),
            other => Err(err(
                id.span(),
                &format!(
                    "unknown SIMD descriptor `{other}` in `#[simd_specialize]`; \
                     expected one of Emu128Descriptor, Sse42Descriptor, AvxDescriptor, \
                     Avx512Descriptor, NeonDescriptor, Simd128Descriptor"
                ),
            )),
        }
    }

    fn try_method(self) -> &'static str {
        match self {
            Self::Emu128 => "try_emu128",
            Self::Sse42 => "try_sse42",
            Self::Avx => "try_avx",
            Self::Avx512 => "try_avx512",
            Self::Neon => "try_neon",
            Self::Simd128 => "try_simd128",
        }
    }

    fn gate_macro(self) -> Option<&'static str> {
        match self {
            Self::Emu128 => None,
            Self::Sse42 => Some("__if_sse42"),
            Self::Avx => Some("__if_avx"),
            Self::Avx512 => Some("__if_avx512"),
            Self::Neon => Some("__if_neon"),
            Self::Simd128 => Some("__if_simd128"),
        }
    }
}

struct Arm {
    isa: Isa,
    target: Vec<TokenTree>,
}

fn parse_arms(attr: TokenStream) -> Result<Vec<Arm>, TokenStream> {
    let toks = flatten_none(attr);
    if toks.is_empty() {
        return Err(err(
            Span::call_site(),
            "`#[simd_specialize]` requires at least one `Descriptor => target_fn` arm",
        ));
    }
    let mut arms = Vec::new();
    let mut seen_mask = 0u8;
    for raw in split_top_level_commas(&toks) {
        let [
            TokenTree::Ident(descr),
            TokenTree::Punct(eq),
            TokenTree::Punct(gt),
            rest @ ..,
        ] = raw
        else {
            return Err(err(
                span_of(raw),
                "expected `Descriptor => target_fn` arm in `#[simd_specialize]`",
            ));
        };
        if eq.as_char() != '=' || eq.spacing() != Spacing::Joint || gt.as_char() != '>' {
            return Err(err(eq.span(), "expected `=>` after descriptor name"));
        }
        if rest.is_empty() {
            return Err(err(
                span_of(raw),
                "expected target function path after `=>`",
            ));
        }
        let isa = Isa::from_ident(descr)?;
        let bit = 1u8 << (isa as u8);
        if seen_mask & bit != 0 {
            return Err(err(
                descr.span(),
                &format!("duplicate `#[simd_specialize]` arm for `{descr}`"),
            ));
        }
        seen_mask |= bit;
        arms.push(Arm {
            isa,
            target: rest.to_vec(),
        });
    }
    Ok(arms)
}

fn quote_ts(span: Span, template: &str, slots: &[(&str, &[TokenTree])]) -> Vec<TokenTree> {
    fn subst(stream: TokenStream, span: Span, slots: &[(&str, &[TokenTree])]) -> Vec<TokenTree> {
        let mut out = Vec::new();
        for tt in stream {
            match tt {
                TokenTree::Group(g) => {
                    let mut ng = Group::new(g.delimiter(), ts(subst(g.stream(), span, slots)));
                    ng.set_span(span);
                    out.push(TokenTree::Group(ng));
                }
                TokenTree::Ident(id) => {
                    let s = id.to_string();
                    if let Some((_, rep)) = slots.iter().find(|(k, _)| *k == s) {
                        out.extend(rep.iter().cloned());
                    } else {
                        let mut nid = id;
                        nid.set_span(span);
                        out.push(TokenTree::Ident(nid));
                    }
                }
                mut other => {
                    other.set_span(span);
                    out.push(other);
                }
            }
        }
        out
    }
    subst(template.parse::<TokenStream>().unwrap(), span, slots)
}

fn ident_tok(name: &str, span: Span) -> Vec<TokenTree> {
    vec![TokenTree::Ident(Ident::new(name, span))]
}

fn punct_tok(ch: char, spacing: Spacing, span: Span) -> TokenTree {
    let mut p = Punct::new(ch, spacing);
    p.set_span(span);
    TokenTree::Punct(p)
}

fn comma_join(items: impl IntoIterator<Item = Vec<TokenTree>>, span: Span) -> Vec<TokenTree> {
    let mut out = Vec::new();
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            out.push(punct_tok(',', Spacing::Alone, span));
        }
        out.extend(item);
    }
    out
}

fn tuple_group(items: &[Vec<TokenTree>], span: Span) -> Vec<TokenTree> {
    let mut inner = Vec::new();
    for item in items {
        inner.extend(item.iter().cloned());
        inner.push(punct_tok(',', Spacing::Alone, span));
    }
    let mut g = Group::new(Delimiter::Parenthesis, ts(inner));
    g.set_span(span);
    vec![TokenTree::Group(g)]
}

fn angle_wrap(items: &[Vec<TokenTree>], span: Span) -> Vec<TokenTree> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut out = vec![punct_tok('<', Spacing::Alone, span)];
    out.extend(comma_join(items.iter().cloned(), span));
    out.push(punct_tok('>', Spacing::Alone, span));
    out
}

fn lifetime_toks(id: &Ident, span: Span) -> Vec<TokenTree> {
    vec![
        punct_tok('\'', Spacing::Joint, span),
        TokenTree::Ident(id.clone()),
    ]
}

fn is_bare_ident(toks: &[TokenTree], i: usize) -> bool {
    let is_colon2 = i >= 2 && is_punct(&toks[i - 2], ':') && is_punct(&toks[i - 1], ':');
    let is_lt = i >= 1 && is_punct(&toks[i - 1], '\'');
    !is_colon2 && !is_lt
}

fn tokens_mention(toks: &[TokenTree], name: &str, lifetime: bool) -> bool {
    for (i, tt) in toks.iter().enumerate() {
        match tt {
            TokenTree::Group(g) => {
                if tokens_mention(&flatten_none(g.stream()), name, lifetime) {
                    return true;
                }
            }
            TokenTree::Ident(id)
                if id.to_string() == name
                    && (if lifetime {
                        i >= 1 && is_punct(&toks[i - 1], '\'')
                    } else {
                        is_bare_ident(toks, i)
                    }) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn tokens_mention_gp(toks: &[TokenTree], gp: &GenericParam<'_>) -> bool {
    match gp.kind {
        GenericKind::Lifetime(id) => tokens_mention(toks, &id.to_string(), true),
        GenericKind::Type(id) | GenericKind::Const(id) => {
            tokens_mention(toks, &id.to_string(), false)
        }
    }
}

fn replace_ident(toks: &[TokenTree], name: &str, replacement: &[TokenTree]) -> Vec<TokenTree> {
    let mut out = Vec::with_capacity(toks.len());
    for (i, tt) in toks.iter().enumerate() {
        match tt {
            TokenTree::Group(g) => {
                let inner = replace_ident(&flatten_none(g.stream()), name, replacement);
                let mut new_g = Group::new(g.delimiter(), ts(inner));
                new_g.set_span(g.span());
                out.push(TokenTree::Group(new_g));
            }
            TokenTree::Ident(id) if is_bare_ident(toks, i) && id.to_string() == name => {
                out.extend(replacement.iter().cloned());
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn desugar_elided(
    toks: &[TokenTree],
    span: Span,
    alloc_lt: &mut impl FnMut() -> Option<Ident>,
) -> Vec<TokenTree> {
    let mut out = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            TokenTree::Group(g) if g.delimiter() != Delimiter::Brace => {
                let inner = desugar_elided(&flatten_none(g.stream()), span, alloc_lt);
                let mut new_g = Group::new(g.delimiter(), ts(inner));
                new_g.set_span(g.span());
                out.push(TokenTree::Group(new_g));
            }
            TokenTree::Punct(p)
                if p.as_char() == '&' && !toks.get(i + 1).is_some_and(|t| is_punct(t, '\'')) =>
            {
                out.push(toks[i].clone());
                if let Some(lt) = alloc_lt() {
                    out.extend(lifetime_toks(&lt, span));
                }
            }
            TokenTree::Punct(p)
                if p.as_char() == '\''
                    && matches!(toks.get(i + 1), Some(TokenTree::Ident(id)) if id.to_string() == "_") =>
            {
                if let Some(lt) = alloc_lt() {
                    out.extend(lifetime_toks(&lt, span));
                    i += 2;
                    continue;
                }
                out.push(toks[i].clone());
            }
            other => out.push(other.clone()),
        }
        i += 1;
    }
    out
}

fn count_elided_lifetimes(toks: &[TokenTree]) -> usize {
    let mut count = 0;
    desugar_elided(toks, Span::call_site(), &mut || {
        count += 1;
        None
    });
    count
}

enum FamilySpec {
    Unit,
    Custom {
        def_and_impl: Vec<TokenTree>,
        callsite_ty: Vec<TokenTree>,
        ret_mentions_d: bool,
    },
}

fn build_family_spec(
    parsed: &ParsedSimdFn<'_>,
    descr_ty_str: &str,
    indep_tys: &[&[TokenTree]],
    dep_tys: &[&[TokenTree]],
    ret_ty_toks: &[TokenTree],
) -> FamilySpec {
    let span = parsed.span;
    let ret_mentions_d =
        !ret_ty_toks.is_empty() && tokens_mention(ret_ty_toks, descr_ty_str, false);
    if dep_tys.is_empty() && !ret_mentions_d {
        return FamilySpec::Unit;
    }

    let indep_elided: usize = indep_tys.iter().map(|ty| count_elided_lifetimes(ty)).sum();
    let mut synth_lts = Vec::new();
    let desugared_dep: Vec<Vec<TokenTree>> = dep_tys
        .iter()
        .map(|ty| {
            desugar_elided(ty, span, &mut || {
                let lt = Ident::new(&format!("__simd_lt{}", synth_lts.len()), span);
                synth_lts.push(lt.clone());
                Some(lt)
            })
        })
        .collect();

    let fn_lts: Vec<&Ident> = parsed
        .generic_params
        .iter()
        .filter_map(|gp| match gp.kind {
            GenericKind::Lifetime(id) if id.to_string() != "static" => Some(id),
            _ => None,
        })
        .collect();

    let desugared_ret = if !ret_mentions_d {
        Vec::new()
    } else if synth_lts.len() == 1 && indep_elided == 0 && fn_lts.is_empty() {
        let lt = synth_lts[0].clone();
        desugar_elided(ret_ty_toks, span, &mut || Some(lt.clone()))
    } else if synth_lts.is_empty() && indep_elided == 0 && fn_lts.len() == 1 {
        let lt = fn_lts[0];
        desugar_elided(ret_ty_toks, span, &mut || Some(lt.clone()))
    } else if synth_lts.is_empty() && fn_lts.is_empty() && indep_elided == 1 {
        let lt = Ident::new("__simd_ret_lt", span);
        let mut used = false;
        let res = desugar_elided(ret_ty_toks, span, &mut || {
            used = true;
            Some(lt.clone())
        });
        if used {
            synth_lts.push(lt);
        }
        res
    } else {
        ret_ty_toks.to_vec()
    };

    let non_d: Vec<(usize, &GenericParam<'_>)> = parsed
        .generic_params
        .iter()
        .enumerate()
        .filter(
            |(_, gp)| !matches!(gp.kind, GenericKind::Type(id) if id.to_string() == descr_ty_str),
        )
        .collect();
    let kept_where: Vec<&[TokenTree]> = parsed
        .where_preds
        .iter()
        .copied()
        .filter(|pred| !bounded_ty_starts_with(pred, descr_ty_str))
        .collect();

    let mut used_in_sig = vec![false; parsed.generic_params.len()];
    for (idx, gp) in &non_d {
        if desugared_dep.iter().any(|ty| tokens_mention_gp(ty, gp))
            || (ret_mentions_d && tokens_mention_gp(&desugared_ret, gp))
        {
            used_in_sig[*idx] = true;
        }
    }
    let mut used_gp = used_in_sig.clone();
    let mut used_where = vec![false; kept_where.len()];
    loop {
        let mut changed = false;
        for (idx, gp) in &non_d {
            if used_gp[*idx] {
                for (idx2, gp2) in &non_d {
                    if !used_gp[*idx2] && tokens_mention_gp(gp.tokens, gp2) {
                        used_gp[*idx2] = true;
                        changed = true;
                    }
                }
            }
        }
        for (widx, pred) in kept_where.iter().enumerate() {
            if !used_where[widx] {
                let lhs = find_single_colon(pred).map_or(*pred, |c| &pred[..c]);
                if non_d
                    .iter()
                    .any(|(idx, gp)| used_gp[*idx] && tokens_mention_gp(lhs, gp))
                {
                    used_where[widx] = true;
                    changed = true;
                }
            }
            if used_where[widx] {
                for (idx2, gp2) in &non_d {
                    if !used_gp[*idx2] && tokens_mention_gp(pred, gp2) {
                        used_gp[*idx2] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let (mut def_params, mut family_args, mut callsite_args) = (Vec::new(), Vec::new(), Vec::new());
    let under_lt = lifetime_toks(&Ident::new("_", span), span);
    for lt in &synth_lts {
        let ltt = lifetime_toks(lt, span);
        def_params.push(ltt.clone());
        family_args.push(ltt);
        callsite_args.push(under_lt.clone());
    }
    for (idx, gp) in &non_d {
        if used_gp[*idx]
            && let GenericKind::Lifetime(id) = gp.kind
        {
            def_params.push(gp.tokens.to_vec());
            family_args.push(lifetime_toks(id, span));
            callsite_args.push(under_lt.clone());
        }
    }
    for (idx, gp) in &non_d {
        if used_gp[*idx]
            && let GenericKind::Type(id) | GenericKind::Const(id) = gp.kind
        {
            def_params.push(gp.tokens.to_vec());
            family_args.push(vec![TokenTree::Ident(id.clone())]);
            callsite_args.push(vec![TokenTree::Ident(id.clone())]);
        }
    }

    let emu_ty = path_tokens(span, &["jxl_simd", "Emu128Descriptor"]);
    let simd_d_tok = ident_tok("__SimdD", span);
    let mut struct_where_preds = Vec::new();
    let mut impl_where_preds = Vec::new();
    for (widx, pred) in kept_where.iter().enumerate() {
        if used_where[widx] {
            if tokens_mention(pred, descr_ty_str, false) {
                impl_where_preds.push(replace_ident(pred, descr_ty_str, &simd_d_tok));
            } else {
                struct_where_preds.push(pred.to_vec());
                impl_where_preds.push(pred.to_vec());
            }
        }
    }

    let fmt_where = |preds: Vec<Vec<TokenTree>>| -> Vec<TokenTree> {
        if preds.is_empty() {
            Vec::new()
        } else {
            let joined = comma_join(preds, span);
            quote_ts(span, "where PREDS", &[("PREDS", &joined)])
        }
    };

    let def_generics = angle_wrap(&def_params, span);
    let struct_fields = if def_params.is_empty() {
        Vec::new()
    } else {
        let mut phantom_inputs: Vec<Vec<TokenTree>> = desugared_dep
            .iter()
            .map(|ty| replace_ident(ty, descr_ty_str, &emu_ty))
            .collect();
        for (idx, gp) in &non_d {
            if used_gp[*idx] && !used_in_sig[*idx] {
                match gp.kind {
                    GenericKind::Lifetime(id) => phantom_inputs.push(quote_ts(
                        span,
                        "&LT ()",
                        &[("LT", &lifetime_toks(id, span))],
                    )),
                    GenericKind::Type(id) => phantom_inputs.push(quote_ts(
                        span,
                        "*const T",
                        &[("T", &[TokenTree::Ident(id.clone())])],
                    )),
                    GenericKind::Const(_) => {}
                }
            }
        }
        let phantom_tuple = tuple_group(&phantom_inputs, span);
        let phantom_ret = if ret_mentions_d {
            replace_ident(&desugared_ret, descr_ty_str, &emu_ty)
        } else {
            tuple_group(&[], span)
        };
        let struct_where = fmt_where(struct_where_preds);
        quote_ts(
            span,
            "(::core::marker::PhantomData<fn((IN_TUPLE)) -> OUT_TY>) WHERE_CL",
            &[
                ("IN_TUPLE", &phantom_tuple),
                ("OUT_TY", &phantom_ret),
                ("WHERE_CL", &struct_where),
            ],
        )
    };

    let mut impl_params = def_params;
    impl_params.push(quote_ts(span, "__SimdD: ::jxl_simd::SimdDescriptor", &[]));
    let impl_generics = angle_wrap(&impl_params, span);
    let family_args_ang = angle_wrap(&family_args, span);
    let impl_where = fmt_where(impl_where_preds);

    let args_assoc = tuple_group(
        &desugared_dep
            .iter()
            .map(|ty| replace_ident(ty, descr_ty_str, &simd_d_tok))
            .collect::<Vec<_>>(),
        span,
    );
    let ret_assoc = if ret_mentions_d {
        replace_ident(&desugared_ret, descr_ty_str, &simd_d_tok)
    } else {
        ident_tok("__SimdIndRet", span)
    };

    let def_and_impl = quote_ts(
        span,
        "#[allow(dead_code, clippy::type_complexity)] struct __SimdFamily DEF_GEN STRUCT_FIELDS; \
         impl IMPL_GEN ::jxl_simd::TypeFamily<__SimdD> for __SimdFamily FAM_ARGS IMPL_WHERE { \
             type Args = ARGS_ASSOC; \
             type Ret<__SimdIndRet> = RET_ASSOC; \
         }",
        &[
            ("DEF_GEN", &def_generics),
            ("STRUCT_FIELDS", &struct_fields),
            ("IMPL_GEN", &impl_generics),
            ("FAM_ARGS", &family_args_ang),
            ("IMPL_WHERE", &impl_where),
            ("ARGS_ASSOC", &args_assoc),
            ("RET_ASSOC", &ret_assoc),
        ],
    );

    let callsite_ang = angle_wrap(&callsite_args, span);
    let callsite_ty = quote_ts(span, "__SimdFamily ARGS", &[("ARGS", &callsite_ang)]);

    FamilySpec::Custom {
        def_and_impl,
        callsite_ty,
        ret_mentions_d,
    }
}

pub(crate) fn simd_specialize_impl(
    attr: TokenStream,
    item: TokenStream,
) -> Result<TokenStream, TokenStream> {
    let arms = parse_arms(attr)?;
    let item_toks = flatten_none(item);
    let parsed = parse_simd_fn(&item_toks, "simd_specialize")?;
    let span = parsed.span;
    let descr_ty_str = parsed.descr_ty_ident.to_string();
    let ret_ty_toks = parsed.ret_toks.get(2..).unwrap_or(&[]);

    let mut outer_params = comma_join(
        [quote_ts(
            span,
            "ATTRS __simd_d: TY",
            &[
                (
                    "ATTRS",
                    &parsed.params[0]
                        .attrs
                        .iter()
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                ("TY", &parsed.params[0].ty),
            ],
        )],
        span,
    );

    enum Bucket {
        Indep(usize),
        Dep(usize),
    }
    let (mut indep_tys, mut indep_exprs, mut indep_pats) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dep_tys, mut dep_exprs, mut dep_pats) = (Vec::new(), Vec::new(), Vec::new());
    let mut param_map = Vec::with_capacity(parsed.params.len() - 1);

    for (idx, param) in parsed.params.iter().skip(1).enumerate() {
        let syn_id = ident_tok(&format!("__simd_arg_{idx}"), span);
        outer_params.push(punct_tok(',', Spacing::Alone, span));
        outer_params.extend(quote_ts(
            span,
            "ATTRS ARG: TY",
            &[
                (
                    "ATTRS",
                    &param.attrs.iter().flatten().cloned().collect::<Vec<_>>(),
                ),
                ("ARG", &syn_id),
                ("TY", &param.ty),
            ],
        ));
        if tokens_mention(&param.ty, &descr_ty_str, false) {
            param_map.push(Bucket::Dep(dep_tys.len()));
            dep_tys.push(param.ty.as_slice());
            dep_exprs.push(syn_id);
            dep_pats.push(param.pat.clone());
        } else {
            param_map.push(Bucket::Indep(indep_tys.len()));
            indep_tys.push(param.ty.as_slice());
            indep_exprs.push(syn_id);
            indep_pats.push(param.pat.clone());
        }
    }

    let (family_defs, family_ty, ind_ret_ty) =
        match build_family_spec(&parsed, &descr_ty_str, &indep_tys, &dep_tys, ret_ty_toks) {
            FamilySpec::Unit => (Vec::new(), tuple_group(&[], span), ident_tok("_", span)),
            FamilySpec::Custom {
                def_and_impl,
                callsite_ty,
                ret_mentions_d,
            } => (
                def_and_impl,
                callsite_ty,
                if ret_mentions_d {
                    tuple_group(&[], span)
                } else {
                    ident_tok("_", span)
                },
            ),
        };

    let tf_elems: Vec<Vec<TokenTree>> = parsed
        .generic_params
        .iter()
        .filter_map(|gp| match gp.kind {
            GenericKind::Type(id) if id.to_string() != descr_ty_str => {
                Some(vec![TokenTree::Ident(id.clone())])
            }
            GenericKind::Const(id) => Some(vec![TokenTree::Ident(id.clone())]),
            _ => None,
        })
        .collect();
    let target_tf = if tf_elems.is_empty() {
        Vec::new()
    } else {
        let ang = angle_wrap(&tf_elems, span);
        quote_ts(span, "::ANG", &[("ANG", &ang)])
    };

    let p_closure: Vec<Vec<TokenTree>> = (0..indep_tys.len())
        .map(|i| ident_tok(&format!("__p{i}"), span))
        .collect();
    let d_closure: Vec<Vec<TokenTree>> = (0..dep_tys.len())
        .map(|i| ident_tok(&format!("__d{i}"), span))
        .collect();
    let p_pats = tuple_group(&p_closure, span);
    let d_pats = tuple_group(&d_closure, span);

    let mut arm_toks = Vec::new();
    for arm in arms {
        let mut target_args = vec![ident_tok("__isa_d", span)];
        for b in &param_map {
            match b {
                Bucket::Indep(pi) => target_args.push(p_closure[*pi].clone()),
                Bucket::Dep(di) => target_args.push(d_closure[*di].clone()),
            }
        }
        let tf = if arm.target.last().is_some_and(|t| is_punct(t, '>')) {
            &[][..]
        } else {
            target_tf.as_slice()
        };
        let args_joined = comma_join(target_args, span);
        let try_fn = ident_tok(arm.isa.try_method(), span);
        let stmt = quote_ts(
            span,
            "let (__simd_p, __simd_da) = match ::jxl_simd::SimdDescriptor::TRY_FN::<FAMILY, _, IND_RET>(\
                 __simd_d, __simd_p, __simd_da, #[inline(always)] |__isa_d, P_PATS, D_PATS| TARGET TF (ARGS)\
             ) { Ok(__simd_ret) => return __simd_ret, Err(__simd_rest) => __simd_rest };",
            &[
                ("TRY_FN", &try_fn),
                ("FAMILY", &family_ty),
                ("IND_RET", &ind_ret_ty),
                ("P_PATS", &p_pats),
                ("D_PATS", &d_pats),
                ("TARGET", &arm.target),
                ("TF", tf),
                ("ARGS", &args_joined),
            ],
        );
        if let Some(gate) = arm.isa.gate_macro() {
            let gate_id = ident_tok(gate, span);
            arm_toks.extend(quote_ts(
                span,
                "::jxl_simd::GATE! { STMT }",
                &[("GATE", &gate_id), ("STMT", &stmt)],
            ));
        } else {
            arm_toks.extend(stmt);
        }
    }

    let orig_where = if parsed.where_preds.is_empty() {
        Vec::new()
    } else {
        let joined = comma_join(parsed.where_preds.iter().map(|p| p.to_vec()), span);
        quote_ts(span, "where PREDS", &[("PREDS", &joined)])
    };
    let orig_generics = angle_wrap(
        &parsed
            .generic_params
            .iter()
            .map(|gp| gp.tokens.to_vec())
            .collect::<Vec<_>>(),
        span,
    );
    let attrs_flat: Vec<TokenTree> = parsed.attrs.into_iter().flatten().collect();
    let body_orig: Vec<TokenTree> = parsed.body_group.stream().into_iter().collect();

    Ok(ts(quote_ts(
        span,
        "ATTRS VIS fn FN_NAME ORIG_GEN (OUTER_PARAMS) RET_TOKS ORIG_WHERE { \
             FAMILY_DEFS \
             let __simd_p = P_EXPRS; \
             let __simd_da = D_EXPRS; \
             ARMS \
             let FIRST_PAT: FIRST_TY = __simd_d; \
             let INDEP_PATS = __simd_p; \
             let DEP_PATS = __simd_da; \
             BODY_ORIG \
         }",
        &[
            ("ATTRS", &attrs_flat),
            ("VIS", parsed.vis),
            ("FN_NAME", &[TokenTree::Ident(parsed.fn_name.clone())]),
            ("ORIG_GEN", &orig_generics),
            ("OUTER_PARAMS", &outer_params),
            ("RET_TOKS", parsed.ret_toks),
            ("ORIG_WHERE", &orig_where),
            ("FAMILY_DEFS", &family_defs),
            ("P_EXPRS", &tuple_group(&indep_exprs, span)),
            ("D_EXPRS", &tuple_group(&dep_exprs, span)),
            ("ARMS", &arm_toks),
            ("FIRST_PAT", &parsed.params[0].pat),
            ("FIRST_TY", &parsed.params[0].ty),
            ("INDEP_PATS", &tuple_group(&indep_pats, span)),
            ("DEP_PATS", &tuple_group(&dep_pats, span)),
            ("BODY_ORIG", &body_orig),
        ],
    )))
}
