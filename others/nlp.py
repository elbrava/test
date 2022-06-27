from math import floor, ceil

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords


def word_return(text):
    words = [i for i in nltk.word_tokenize(text) if not (len(i) == 1 and not i.isalnum())]
    words = [w for w in words if w.lower() not in set(stopwords.words("english"))]
    words = FreqDist(word.lower() for word in words)
    return words.most_common()


def main():
    p = word_return(
        r'''rem ipsum dolor sit amet, consecrate disciplining elit. Risque nil eros,
    pulmonary facilitates justo mollies, actor consequent urn. Morbid a biennium metes.
    Done squelchier solicitudes enum eu venation. Dis incident lorem ex,
    premium orc vestibule wget. Class aptest tacit sociology ad littoral torque
    per connubial nostril, per inceptions mimenames. Dis pharaoh quits lacks ut
    vestibule. Nascence ipsum lacks, niacin quits postgres ut, pulmonary vitae dolor.
    Integer eu nib love "love" at nisei perambulator sagittal id vel leo. Integer fugitive
    faucets libero, at maximus nil suspicious postgres. Morbid nec enum nun.
    Phallus biennium turps ut ipsum gestates, sed solicitudes elit collision.
    Crash pharaoh mi critique sapien vestibule lobotomist. Nam wget biennium metes,
    non dictum Lauris. Null at tells sagittal, riviera est a, referendum
    '''*1200)
    t = ceil((len(p) * 0.03) // 100)
    if t == 0: t = 3
    print(t)
    print(p[:t])


if __name__ == '__main__':
    main()
