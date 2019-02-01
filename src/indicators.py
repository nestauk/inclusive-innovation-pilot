import sys
import pandas as pd


class Indicators():

    def __init__(self, data):
        self.data = data

    def people_diversity(self, *args, thresh=25):
        """Find the gender / ethnic diversity of the people that are currently
        working in an area.

        Args:
            thresh (:obj:`int`): Filter out instances with a count lower than
                the threshold.
            *args: Usually location (city, country, continent), gender or
                ethnicity and degree type. Note that the first argument is used
                to normalise the data.

        Return:
            div (:obj:`pandas.DataFrame`): A DataFrame grouped by the args
                and reindexed based on the number of women/men in the cities.

        """
        df = self.data[(self.data.is_current == 1)
                       & (self.data.primary_role == 'company')] \
                 .drop_duplicates('person_id')
        city_gender_pop = df.groupby(list(args)).count()['person_id']
        city_pop = df.groupby(args[0]).count()['person_id']
        idx = city_pop.where(city_pop > thresh).dropna() \
                      .sort_values(ascending=False).index
        div = city_gender_pop / city_pop
        return div.reindex(idx, level=0)

    def city_role_company(self, *args):
        """Count the number of women/men in each job type and every company
            size.

        Args:
            *args: Usually location (city, country, continent), gender or
                ethnicity, company size and job type.

        Return:
            (:obj:`pandas.DataFrame`): A DataFrame grouped by the args.

        """
        return self.data[self.data.primary_role == 'company'] \
                   .drop_duplicates('person_id') \
                   .groupby(list(args)) \
                   .count()['person_id']

    def home_study(self):
        home_study = self.data[(self.data.primary_role == 'university')
                               & (self.data.org_id
                                      .isin(self.data.institution_id.dropna()
                                                     .unique()))] \
                                   .drop_duplicates('person_id').shape[0]
        all_universities = self.data[self.data.primary_role == 'university'] \
                               .unique().shape[0]
        return home_study / all_universities


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


def main():
    data = pd.read_csv(sys.argv[1])
    indicators = Indicators(data)

    print('CITY -- GENDER')
    gender_diversity = indicators.people_diversity('city', 'gender')
    print(gender_diversity)
    print()

    print('DEGREE -- GENDER')
    degree_diversity = indicators.people_diversity('degree_type', 'gender')
    print(degree_diversity)

    print('GENDER -- DEGREE')
    gd_div = indicators.people_diversity('gender', 'degree_type')
    print(gd_div.where(gd_div > 0.02).dropna())
    print()

    print('CITY -- JOB TYPE -- GENDER')
    role_comp_div = indicators.city_role_company('city', 'job_type', 'gender')
    print(role_comp_div)
    print()

    print('CITY -- CATEGORY_GROUP_LIST -- GENDER')
    cat_comp_div = indicators.city_role_company('city',
                                                'category_group_list',
                                                'gender')
    print(cat_comp_div)
    print()


if __name__ == '__main__':
    main()
