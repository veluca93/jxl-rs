// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::iter::FromIterator;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

fn err(span: Span, msg: &str) -> TokenStream {
    let mut inner = TokenStream::new();
    let mut lit = Literal::string(msg);
    lit.set_span(span);
    inner.extend([TokenTree::Literal(lit)]);
    let mut call = Group::new(Delimiter::Parenthesis, inner);
    call.set_span(span);
    let mut out = TokenStream::new();
    out.extend([
        TokenTree::Ident(Ident::new("compile_error", span)),
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        TokenTree::Group(call),
        TokenTree::Punct(Punct::new(';', Spacing::Alone)),
    ]);
    out
}

fn ts(trees: impl IntoIterator<Item = TokenTree>) -> TokenStream {
    TokenStream::from_iter(trees)
}

fn flatten_none(stream: TokenStream) -> Vec<TokenTree> {
    fn go(stream: TokenStream, out: &mut Vec<TokenTree>) {
        for tt in stream {
            if let TokenTree::Group(g) = &tt
                && g.delimiter() == Delimiter::None
            {
                go(g.stream(), out);
            } else {
                out.push(tt);
            }
        }
    }
    let mut out = Vec::new();
    go(stream, &mut out);
    out
}

fn is_ident(tt: &TokenTree, kw: &str) -> bool {
    matches!(tt, TokenTree::Ident(id) if id.to_string() == kw)
}

fn is_punct(tt: &TokenTree, ch: char) -> bool {
    matches!(tt, TokenTree::Punct(p) if p.as_char() == ch)
}

fn span_of(toks: &[TokenTree]) -> Span {
    toks.first().map_or_else(Span::call_site, TokenTree::span)
}

fn path_tokens(span: Span, segments: &[&str]) -> Vec<TokenTree> {
    let mut out = Vec::new();
    for seg in segments {
        out.push(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
        out.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
        out.push(TokenTree::Ident(Ident::new(seg, span)));
    }
    out
}

fn paren(span: Span, inner: TokenStream) -> TokenTree {
    let mut g = Group::new(Delimiter::Parenthesis, inner);
    g.set_span(span);
    TokenTree::Group(g)
}

fn bracket(span: Span, inner: TokenStream) -> TokenTree {
    let mut g = Group::new(Delimiter::Bracket, inner);
    g.set_span(span);
    TokenTree::Group(g)
}

fn brace(span: Span, inner: TokenStream) -> TokenTree {
    let mut g = Group::new(Delimiter::Brace, inner);
    g.set_span(span);
    TokenTree::Group(g)
}

fn attr_first_ident(attr_toks: &[TokenTree]) -> Option<String> {
    let [_, TokenTree::Group(g)] = attr_toks else {
        return None;
    };
    let inner = flatten_none(g.stream());
    match inner.first() {
        Some(TokenTree::Ident(id)) => Some(id.to_string()),
        _ => None,
    }
}

fn parse_outer_attrs(toks: &[TokenTree], pos: &mut usize) -> Vec<Vec<TokenTree>> {
    let mut attrs = Vec::new();
    while *pos + 1 < toks.len()
        && is_punct(&toks[*pos], '#')
        && matches!(&toks[*pos + 1], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket)
    {
        attrs.push(toks[*pos..*pos + 2].to_vec());
        *pos += 2;
    }
    attrs
}

fn split_top_level_commas(toks: &[TokenTree]) -> Vec<&[TokenTree]> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut angle_depth = 0usize;
    let mut i = 0;
    while i < toks.len() {
        if is_arrow(toks, i) {
            i += 2;
            continue;
        }
        match &toks[i] {
            TokenTree::Punct(p) if p.as_char() == '<' => angle_depth += 1,
            TokenTree::Punct(p) if p.as_char() == '>' => {
                angle_depth = angle_depth.saturating_sub(1)
            }
            TokenTree::Punct(p) if p.as_char() == ',' && angle_depth == 0 => {
                if start < i {
                    parts.push(&toks[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < toks.len() {
        parts.push(&toks[start..]);
    }
    parts
}

fn find_single_colon(toks: &[TokenTree]) -> Option<usize> {
    let mut angle_depth = 0usize;
    let mut i = 0;
    while i < toks.len() {
        if is_arrow(toks, i) {
            i += 2;
            continue;
        }
        if is_punct(&toks[i], '<') {
            angle_depth += 1;
        } else if is_punct(&toks[i], '>') {
            angle_depth = angle_depth.saturating_sub(1);
        } else if is_punct(&toks[i], ':') {
            if toks.get(i + 1).is_some_and(|n| is_punct(n, ':')) {
                i += 2;
                continue;
            }
            if angle_depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn default_inline_attr(has_inline: bool, span: Span) -> TokenStream {
    if has_inline {
        return TokenStream::new();
    }
    let body: TokenStream = [
        TokenTree::Ident(Ident::new("cfg_attr", span)),
        paren(
            span,
            "any(target_arch = \"x86_64\", target_arch = \"aarch64\", target_arch = \"wasm32\"), inline(always)"
                .parse()
                .unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    ts([
        TokenTree::Punct(Punct::new('#', Spacing::Alone)),
        bracket(span, body),
    ])
}

enum GenericKind<'a> {
    Lifetime,
    Type(&'a Ident),
    Const(&'a Ident),
}

struct GenericParam<'a> {
    tokens: &'a [TokenTree],
    kind: GenericKind<'a>,
}

fn parse_generics<'a>(
    toks: &'a [TokenTree],
    pos: &mut usize,
) -> Result<Vec<GenericParam<'a>>, TokenStream> {
    if !toks.get(*pos).is_some_and(|t| is_punct(t, '<')) {
        return Ok(Vec::new());
    }
    let open_span = toks[*pos].span();
    let start = *pos + 1;
    let mut depth = 1usize;
    let mut i = start;
    while i < toks.len() {
        if is_arrow(toks, i) {
            i += 2;
            continue;
        }
        if is_punct(&toks[i], '<') {
            depth += 1;
        } else if is_punct(&toks[i], '>') {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        i += 1;
    }
    if depth != 0 {
        return Err(err(open_span, "unclosed `<` in generic parameter list"));
    }
    *pos = i + 1;

    let mut params = Vec::new();
    for raw in split_top_level_commas(&toks[start..i]) {
        if is_punct(&raw[0], '\'') {
            params.push(GenericParam {
                tokens: raw,
                kind: GenericKind::Lifetime,
            });
        } else if is_ident(&raw[0], "const") {
            let Some(TokenTree::Ident(id)) = raw.get(1) else {
                return Err(err(span_of(raw), "expected identifier after `const`"));
            };
            params.push(GenericParam {
                tokens: raw,
                kind: GenericKind::Const(id),
            });
        } else if let TokenTree::Ident(id) = &raw[0] {
            params.push(GenericParam {
                tokens: raw,
                kind: GenericKind::Type(id),
            });
        } else {
            return Err(err(span_of(raw), "unsupported generic parameter"));
        }
    }
    Ok(params)
}

fn bounded_ty_starts_with(pred: &[TokenTree], ident_str: &str) -> bool {
    // Strip optional `for<'a>` binder before checking the bounded type.
    let mut slice = pred;
    if slice.first().is_some_and(|t| is_ident(t, "for")) {
        slice = &slice[1..];
        if slice.first().is_some_and(|t| is_punct(t, '<')) {
            let mut depth = 1usize;
            let mut j = 1;
            while j < slice.len() && depth > 0 {
                if is_punct(&slice[j], '<') {
                    depth += 1;
                } else if is_punct(&slice[j], '>') {
                    depth -= 1;
                }
                j += 1;
            }
            slice = &slice[j..];
        }
    }
    match slice.first() {
        Some(TokenTree::Ident(id)) => id.to_string() == ident_str,
        _ => false,
    }
}

struct ParsedParam {
    attrs: Vec<Vec<TokenTree>>,
    pat: Vec<TokenTree>,
    ty: Vec<TokenTree>,
}

fn parse_params(stream: TokenStream, err_span: Span) -> Result<Vec<ParsedParam>, TokenStream> {
    let toks = flatten_none(stream);
    let mut out = Vec::new();
    for raw in split_top_level_commas(&toks) {
        let mut idx = 0;
        let attrs = parse_outer_attrs(raw, &mut idx);
        let rest = &raw[idx..];
        let Some(colon) = find_single_colon(rest) else {
            return Err(err(
                span_of(rest.first().map_or(&[], |x| std::slice::from_ref(x))),
                "expected typed parameter `name: Type` (methods taking `self` are not supported)",
            ));
        };
        let pat = rest[..colon].to_vec();
        let ty = flatten_none(ts(rest[colon + 1..].iter().cloned()));
        if ty.is_empty() {
            return Err(err(err_span, "expected parameter type after `:`"));
        }
        out.push(ParsedParam { attrs, pat, ty });
    }
    Ok(out)
}

fn clean_param_ident(pat: &[TokenTree], idx: usize, fallback_span: Span) -> Ident {
    match pat {
        [TokenTree::Ident(id)] => id.clone(),
        [mut_kw, TokenTree::Ident(id)] if is_ident(mut_kw, "mut") => id.clone(),
        _ => Ident::new(&format!("__simd_arg_{idx}"), fallback_span),
    }
}

fn is_arrow(toks: &[TokenTree], pos: usize) -> bool {
    matches!(
        (toks.get(pos), toks.get(pos + 1)),
        (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)))
            if a.as_char() == '-' && a.spacing() == Spacing::Joint && b.as_char() == '>'
    )
}

/// Generates a runtime-dispatched wrapper function around a generic SIMD implementation.
///
/// # Example
/// ```ignore
/// #[simd_dispatch(my_op)]
/// fn my_op_simd<D: SimdDescriptor>(d: D, input: &[f32], output: &mut [f32]) {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn simd_dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    match simd_dispatch_impl(attr, item) {
        Ok(tokens) => tokens,
        Err(err_ts) => err_ts,
    }
}

fn simd_dispatch_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream, TokenStream> {
    let attr_toks = flatten_none(attr);
    let [TokenTree::Ident(dname)] = attr_toks.as_slice() else {
        return Err(err(
            span_of(&attr_toks),
            "expected dispatch function identifier, e.g. `#[simd_dispatch(my_fn)]`",
        ));
    };

    let item_toks = flatten_none(item.clone());
    let mut pos = 0;
    let attrs = parse_outer_attrs(&item_toks, &mut pos);
    let has_inline = attrs
        .iter()
        .any(|a| attr_first_ident(a).as_deref() == Some("inline"));

    let vis_start = pos;
    if item_toks.get(pos).is_some_and(|t| is_ident(t, "pub")) {
        pos += 1;
        if matches!(
            item_toks.get(pos),
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis
        ) {
            pos += 1;
        }
    }
    let vis = &item_toks[vis_start..pos];

    if !item_toks.get(pos).is_some_and(|t| is_ident(t, "fn")) {
        return Err(err(
            span_of(&item_toks[pos..]),
            "`#[simd_dispatch]` must be applied to a `fn` item",
        ));
    }
    pos += 1;

    let Some(TokenTree::Ident(fn_name)) = item_toks.get(pos) else {
        return Err(err(span_of(&item_toks[pos..]), "expected function name"));
    };
    let span = fn_name.span();
    pos += 1;

    let generic_params = parse_generics(&item_toks, &mut pos)?;

    let Some(TokenTree::Group(params_group)) = item_toks.get(pos) else {
        return Err(err(span, "expected parameter list"));
    };
    if params_group.delimiter() != Delimiter::Parenthesis {
        return Err(err(params_group.span(), "expected parameter list `(...)`"));
    }
    pos += 1;

    let params = parse_params(params_group.stream(), params_group.span())?;
    let Some(first_param) = params.first() else {
        return Err(err(
            params_group.span(),
            "expected first parameter to be the generic SIMD descriptor, e.g. `d: D`",
        ));
    };
    let [TokenTree::Ident(descr_ty_ident)] = first_param.ty.as_slice() else {
        return Err(err(
            span_of(&first_param.ty),
            "expected first parameter type to be the generic `SimdDescriptor` type parameter (e.g. `d: D`)",
        ));
    };
    let descr_ty_str = descr_ty_ident.to_string();
    if !generic_params
        .iter()
        .any(|gp| matches!(gp.kind, GenericKind::Type(id) if id.to_string() == descr_ty_str))
    {
        return Err(err(
            descr_ty_ident.span(),
            &format!("type `{descr_ty_str}` must be a generic type parameter on `{fn_name}`"),
        ));
    }

    let ret_start = pos;
    if is_arrow(&item_toks, pos) {
        pos += 2;
        while pos < item_toks.len()
            && !is_ident(&item_toks[pos], "where")
            && !matches!(&item_toks[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Brace)
        {
            pos += 1;
        }
    }
    let ret_toks = &item_toks[ret_start..pos];

    let mut kept_where: Vec<&[TokenTree]> = Vec::new();
    if item_toks.get(pos).is_some_and(|t| is_ident(t, "where")) {
        pos += 1;
        let where_start = pos;
        while pos < item_toks.len()
            && !matches!(&item_toks[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Brace)
        {
            pos += 1;
        }
        for pred in split_top_level_commas(&item_toks[where_start..pos]) {
            if !bounded_ty_starts_with(pred, &descr_ty_str) {
                kept_where.push(pred);
            }
        }
    }

    if !matches!(
        item_toks.get(pos),
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace
    ) {
        return Err(err(span_of(&item_toks[pos..]), "expected function body"));
    }

    // Build generic list for the dispatch wrapper (strip `D`), plus turbofish args for the call.
    let mut dispatch_generic_list: Vec<&[TokenTree]> = Vec::new();
    let mut turbofish: Vec<TokenTree> = Vec::new();
    let mut has_extra_type_or_const = false;
    for gp in &generic_params {
        match gp.kind {
            GenericKind::Lifetime => dispatch_generic_list.push(gp.tokens),
            GenericKind::Type(id) if id.to_string() == descr_ty_str => {
                if !turbofish.is_empty() {
                    turbofish.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
                }
                turbofish.push(TokenTree::Ident(Ident::new("_", span)));
            }
            GenericKind::Type(id) | GenericKind::Const(id) => {
                has_extra_type_or_const = true;
                dispatch_generic_list.push(gp.tokens);
                if !turbofish.is_empty() {
                    turbofish.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
                }
                turbofish.push(TokenTree::Ident(id.clone()));
            }
        }
    }

    let mut clean_params_inner = Vec::new();
    let mut arg_vals = Vec::new();
    for (idx, param) in params.iter().skip(1).enumerate() {
        let arg_id = clean_param_ident(&param.pat, idx, span);
        if !clean_params_inner.is_empty() {
            clean_params_inner.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
            arg_vals.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
        }
        for attr in &param.attrs {
            clean_params_inner.extend(attr.iter().cloned());
        }
        clean_params_inner.push(TokenTree::Ident(arg_id.clone()));
        clean_params_inner.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
        clean_params_inner.extend(param.ty.iter().cloned());
        arg_vals.push(TokenTree::Ident(arg_id));
    }

    let descr_arg = Ident::new("__d", span);
    let mut call_toks = Vec::new();
    call_toks.push(TokenTree::Ident(fn_name.clone()));
    if has_extra_type_or_const {
        call_toks.push(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
        call_toks.push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
        call_toks.push(TokenTree::Punct(Punct::new('<', Spacing::Alone)));
        call_toks.extend(turbofish);
        call_toks.push(TokenTree::Punct(Punct::new('>', Spacing::Alone)));
    }
    let mut call_args = vec![TokenTree::Ident(descr_arg.clone())];
    if !arg_vals.is_empty() {
        call_args.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
        call_args.extend(arg_vals);
    }
    call_toks.push(paren(span, ts(call_args)));

    let mut macro_body = vec![
        TokenTree::Punct(Punct::new('|', Spacing::Alone)),
        TokenTree::Ident(descr_arg),
        TokenTree::Punct(Punct::new('|', Spacing::Alone)),
    ];
    macro_body.extend(call_toks);

    let mut fn_body = path_tokens(span, &["jxl_simd", "__simd_dispatch"]);
    fn_body.push(TokenTree::Punct(Punct::new('!', Spacing::Alone)));
    fn_body.push(paren(span, ts(macro_body)));

    let mut out = default_inline_attr(has_inline, span);
    out.extend(item);

    for attr in attrs {
        if attr_first_ident(&attr).as_deref() != Some("inline") {
            out.extend(attr);
        }
    }
    out.extend(vis.iter().cloned());
    out.extend([
        TokenTree::Ident(Ident::new("fn", span)),
        TokenTree::Ident(dname.clone()),
    ]);
    if !dispatch_generic_list.is_empty() {
        out.extend([TokenTree::Punct(Punct::new('<', Spacing::Alone))]);
        for (i, gp) in dispatch_generic_list.iter().enumerate() {
            if i > 0 {
                out.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
            }
            out.extend((*gp).iter().cloned());
        }
        out.extend([TokenTree::Punct(Punct::new('>', Spacing::Alone))]);
    }
    out.extend([paren(span, ts(clean_params_inner))]);
    out.extend(ret_toks.iter().cloned());
    if !kept_where.is_empty() {
        out.extend([TokenTree::Ident(Ident::new("where", span))]);
        for (i, pred) in kept_where.iter().enumerate() {
            if i > 0 {
                out.extend([TokenTree::Punct(Punct::new(',', Spacing::Alone))]);
            }
            out.extend((*pred).iter().cloned());
        }
    }
    out.extend([brace(span, ts(fn_body))]);
    Ok(out)
}

/// Generates per-instruction-set `#[test]` functions calling the annotated generic SIMD test.
///
/// # Example
/// ```ignore
/// #[simd_test]
/// fn test_neg<D: SimdDescriptor>(d: D) {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn simd_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    match simd_test_impl(attr, item) {
        Ok(tokens) => tokens,
        Err(err_ts) => err_ts,
    }
}

fn simd_test_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream, TokenStream> {
    let attr_toks = flatten_none(attr);
    if !attr_toks.is_empty() {
        return Err(err(
            span_of(&attr_toks),
            "`#[simd_test]` does not accept arguments",
        ));
    }

    let item_toks = flatten_none(item);
    let mut pos = 0;
    let attrs = parse_outer_attrs(&item_toks, &mut pos);
    let (test_attrs, kept_attrs): (Vec<_>, Vec<_>) = attrs.into_iter().partition(|a| {
        matches!(
            attr_first_ident(a).as_deref(),
            Some("ignore" | "should_panic")
        )
    });
    let has_inline = kept_attrs
        .iter()
        .any(|a| attr_first_ident(a).as_deref() == Some("inline"));

    let mut sig_pos = pos;
    if item_toks.get(sig_pos).is_some_and(|t| is_ident(t, "pub")) {
        sig_pos += 1;
        if matches!(
            item_toks.get(sig_pos),
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis
        ) {
            sig_pos += 1;
        }
    }
    if !item_toks.get(sig_pos).is_some_and(|t| is_ident(t, "fn")) {
        return Err(err(
            span_of(&item_toks[sig_pos..]),
            "`#[simd_test]` must be applied to a `fn` item",
        ));
    }
    sig_pos += 1;
    let Some(TokenTree::Ident(fn_name)) = item_toks.get(sig_pos) else {
        return Err(err(
            span_of(&item_toks[sig_pos..]),
            "expected test function name",
        ));
    };
    let span = fn_name.span();
    sig_pos += 1;
    let _generics = parse_generics(&item_toks, &mut sig_pos)?;
    let Some(TokenTree::Group(params_group)) = item_toks.get(sig_pos) else {
        return Err(err(span, "expected parameter list"));
    };
    let params = parse_params(params_group.stream(), params_group.span())?;
    if params.len() != 1 {
        return Err(err(
            params_group.span(),
            "`#[simd_test]` function must take a single `d: D` SIMD descriptor parameter",
        ));
    }

    let mut out = TokenStream::new();
    for attr in kept_attrs {
        out.extend(attr);
    }
    out.extend(default_inline_attr(has_inline, span));
    out.extend(item_toks[pos..].iter().cloned());

    let flat_test_attrs: TokenStream = test_attrs.into_iter().flatten().collect();
    let mut macro_args = vec![
        TokenTree::Ident(fn_name.clone()),
        TokenTree::Punct(Punct::new(',', Spacing::Alone)),
        bracket(span, flat_test_attrs),
    ];
    for suffix in ["avx512", "avx", "sse42", "neon", "simd128", "scalar"] {
        macro_args.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
        macro_args.push(TokenTree::Ident(Ident::new(
            &format!("{fn_name}_{suffix}"),
            span,
        )));
    }

    out.extend(path_tokens(span, &["jxl_simd", "__simd_test_fns"]));
    out.extend([
        TokenTree::Punct(Punct::new('!', Spacing::Alone)),
        paren(span, ts(macro_args)),
        TokenTree::Punct(Punct::new(';', Spacing::Alone)),
    ]);
    Ok(out)
}
