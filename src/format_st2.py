"""
Helper functions to convert `text_w_pairs` to BIO format, and back
"""

import re


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub('</*[A-Z]+\d*>','',tok) 


def get_BIO(text_w_pairs):
    tokens = []
    ce_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):

        # Replace if special
        if '<ARG0>' in tok:
            tok = re.sub('<ARG0>','',tok)
            tag = 'B-C'
            next_tag = 'I-C'
        elif '</ARG0>' in tok:
            tok = re.sub('</ARG0>','',tok)
            tag = 'I-C'
            next_tag = 'O'
        elif '<ARG1>' in tok:
            tok = re.sub('<ARG1>','',tok)
            tag = 'B-E'
            next_tag = 'I-E'
        elif '</ARG1>' in tok:
            tok = re.sub('</ARG1>','',tok)
            tag = 'I-E'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        ce_tags.append(tag)
        tag = next_tag
    
    return tokens, ce_tags


def get_BIO_sig(text_w_pairs):
    tokens = []
    s_tags = []
    next_tag = tag = 'O'
    for tok in text_w_pairs.split(' '):
        # Replace if special
        if '<SIG' in tok:
            tok = re.sub('<SIG([A-Z]|\d)*>','',tok)
            tag = 'B-S'
            next_tag = 'I-S'
            if '</SIG' in tok: # one word only
                tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
                next_tag = 'O'

        elif '</SIG' in tok:
            tok = re.sub('</SIG([A-Z]|\d)*>','',tok)
            tag = 'I-S'
            next_tag = 'O'

        tokens.append(clean_tok(tok))
        s_tags.append(tag)
        tag = next_tag
    
    return tokens, s_tags


def get_BIO_all(text_w_pairs, verbose=False):
    tokens, ce_tags = get_BIO(text_w_pairs)
    tokens_s, s_tags = get_BIO_sig(text_w_pairs)
    assert(tokens==tokens_s)
    assert(len(ce_tags)==len(s_tags)==len(tokens))
    if verbose:
        print(f'Tokens: {tokens}')
        print(f'CE Tags: {ce_tags}')
        print(f'Sig Tags: {s_tags}')
    return tokens, ce_tags, s_tags


def get_text_w_pairs(tokens, ce_list, sig_list):
    """
    tokens [list] : White-space separated list of word tokens.
    ce_list [list] : List of Cause-Effect tags. Either in ['C','E','O'] format, or BIO-format ['B-C','I-C'...]
    sig_list [list] : List of Signal tags. Either in ['S','O'] format, or BIO-format ['B-S','I-S'...]
    """
    
    # Sanity check
    assert(len(tokens)==len(ce_list))
    assert(len(ce_list)==len(sig_list))
    
    # Loop per token
    curr_ce = prev_ce = None
    prev_sig = prev_sig = None
    for i, (tok, ce, sig) in enumerate(zip(tokens, ce_list, sig_list)):

        curr_ce = ce.split('-')[-1]
        curr_sig = sig.split('-')[-1]

        if curr_sig!=prev_sig: # we only need to tag BOUNDARIES

            # opening
            if curr_sig=='S':
                tokens[i]='<SIG0>'+tok

            # closing
            if prev_sig=='S':
                tokens[i-1]=tokens[i-1]+'</SIG0>'

        if curr_ce!=prev_ce: # we only need to tag BOUNDARIES

            # opening
            if curr_ce=='C':
                tokens[i]='<ARG0>'+tok
            elif curr_ce=='E':
                tokens[i]='<ARG1>'+tok

            # closing
            if prev_ce=='C':
                tokens[i-1]=tokens[i-1]+'</ARG0>'
            elif prev_ce=='E':
                tokens[i-1]=tokens[i-1]+'</ARG1>'

        # update
        prev_ce = curr_ce
        prev_sig = curr_sig

    # LAST closure
    if prev_sig=='S':
        tokens[i]=tokens[i]+'</SIG0>'

    if prev_ce=='C':
        tokens[i]=tokens[i]+'</ARG0>'
    elif prev_ce=='E':
        tokens[i]=tokens[i]+'</ARG1>'
        
    return ' '.join(tokens)


if __name__ == "__main__":
    # Demo text_w_pairs to BIO format
    print('\n >>> Demo `get_BIO_all`')
    text_w_pairs = '<ARG1>Three people were killed and 69 others injured</ARG1> in <ARG0>the explosion</ARG0> .'
    tokens, ce_ref, sig_ref = get_BIO_all(text_w_pairs, verbose=False)
    print('Input text_w_pairs: ', text_w_pairs)
    print('Output CE List: ', ce_ref)
    print('Output Sig List: ', sig_ref)


    # Demo BIO format to text_w_pairs
    print('\n >>> Demo `get_text_w_pairs`')
    print('Input CE List: ', ce_ref)
    print('Input Sig List: ', sig_ref)
    print('Output text_w_pairs: ', get_text_w_pairs(tokens, ce_ref, sig_ref))

    print('\n >>> Demo `get_text_w_pairs` without BIO format')
    tokens = ['Three', 'people', 'were', 'killed', 'and', '69', 'others', 'injured', 'in', 'the', 'explosion', '.']
    ce_pred = ['O', 'O', 'O', 'O', 'C', 'O', 'E', 'E', 'E', 'E', 'O', 'O']
    sig_pred = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S']
    print('Input CE List: ', ce_pred)
    print('Input Sig List: ', sig_pred)
    print('Output text_w_pairs: ', get_text_w_pairs(tokens, ce_pred, sig_pred))