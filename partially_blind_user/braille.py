
'''# Grade-1 (uncontracted) translator using Unicode Braille Patterns block (U+2800 ..)

LETTER_MASKS = [
    1,   # a
    3,   # b
    9,   # c
    25,  # d
    17,  # e
    11,  # f
    27,  # g
    19,  # h
    10,  # i
    26,  # j
    5,   # k
    7,   # l
    13,  # m
    29,  # n
    21,  # o
    15,  # p
    31,  # q
    23,  # r
    14,  # s
    30,  # t
    69,  # u
    71,  # v
    90,  # w
    77,  # x
    93,  # y
    85   # z
]

BASE = 0x2800
NUMBER_SIGN = chr(BASE + 60)   # dots 3-4-5-6
CAPITAL_SIGN = chr(BASE + 32)  # dot 6
SPACE = chr(BASE + 0)

def letter_to_braille(ch: str) -> str:
    idx = ord(ch.lower()) - ord('a')
    if 0 <= idx < 26:
        return chr(BASE + LETTER_MASKS[idx])
    return '?'  # fallback for unexpected

def text_to_braille(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # handle spaces
        if ch.isspace():
            out.append(SPACE)
            i += 1
            continue

        # handle runs of digits -> prefix number sign once
        if ch.isdigit():
            out.append(NUMBER_SIGN)
            while i < n and text[i].isdigit():
                d = text[i]
                # digits 1-9,0 map to letters a-j (1->a,2->b,...,9->i,0->j)
                if d == '0':
                    mapped = letter_to_braille('j')
                else:
                    mapped = letter_to_braille(chr(ord('a') + int(d) - 1))
                out.append(mapped)
                i += 1
            continue

        # handle letters (possibly uppercase)
        if ch.isalpha():
            if ch.isupper():
                out.append(CAPITAL_SIGN)
            out.append(letter_to_braille(ch))
            i += 1
            continue

        # For other characters (punctuation) - simple fallback: include space or placeholder
        # You can extend this dictionary to add punctuation mappings.
        out.append('?')
        i += 1

    return ''.join(out)

if __name__ == "__main__":
    examples = [
        "Hello World",
        "Hi 123",
        "Braille 2025",
        "aBc XyZ 0"
    ]
    for ex in examples:
        print(f"Text : {ex}")
        print("Braille:", text_to_braille(ex))
        print()
'''

BASE = 0x2800
LETTER_MASKS = [
    1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13, 29, 21, 15,
    31, 23, 14, 30, 69, 71, 90, 77, 93, 85
]
NUMBER_SIGN = chr(BASE + 60)  
CAPITAL_SIGN = chr(BASE + 32)  
SPACE = chr(BASE + 0)

def letter_to_braille(ch: str) -> str:
    idx = ord(ch.lower()) - ord('a')
    if 0 <= idx < 26:
        return chr(BASE + LETTER_MASKS[idx])
    return '?'

def text_to_braille(text: str) -> str:
    """Convert plain English text to Unicode Braille (Grade 1)"""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            out.append(SPACE)
            i += 1
            continue
        if ch.isdigit():
            out.append(NUMBER_SIGN)
            while i < n and text[i].isdigit():
                d = text[i]
                mapped = letter_to_braille('j' if d == '0' else chr(ord('a') + int(d) - 1))
                out.append(mapped)
                i += 1
            continue
        if ch.isalpha():
            if ch.isupper():
                out.append(CAPITAL_SIGN)
            out.append(letter_to_braille(ch))
            i += 1
            continue
        out.append('?')
        i += 1
    return ''.join(out)

