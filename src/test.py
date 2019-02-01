d = {
 1: [.06, .4, .44, .1],
 2: [.39, .39, .22],
 3: [.44, .56],
 4: [.62, .38],
 5: [.39, .61],
 6: [.04, .45, .51]
}


def lieberson_index(d):
    """Measure Lieberson's Aw diversity within a population. Aw receives a set
        of variables V with p categories and uses the proportions Yk in each
        category to measure the diversity of the set.

    Args:
        d (:obj:`dict`): d.keys() contains the variables V. d.values() contains
            lists with the proportions of each category for every variable.

    Return:
        aw (:obj:`float`): Lieberson's Index of diversity.

    """
    yk = sum([sum([v**2 for v in vals]) for vals in d.values()])
    aw = 1 - yk / len(d)
    return aw
