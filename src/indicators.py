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

    def home_study(self, country):
        df = self.data[self.data.country == country]
        local_uni = df[df.institution_id
                         .isin(set(df.org_id)
                               & set(df.institution_id))].person_id \
                                                         .unique().shape[0]

        all_universities = df.person_id.unique().shape[0]
        return (local_uni / all_universities) * 100

    def lieberson_format(self, cols, country_level=False, city_level=False,
                         country=None):
        """Format data for Lieberson index.

        TODO: Country level.

        """
        df = self.data.drop_duplicates('person_id')

        if city_level and country is not None:
            dfs = [df[df.city == city] for city in df.city.unique()]

        # if country_level:
        #     df = df[df.country == country]

        city_level_format = {}
        for df in dfs:
            d = {}
            for col in cols:
                d[col] = list(df[col].value_counts() / df.shape[0])
            city_level_format[df['city'].unique()[0]] = d
        return city_level_format

    def lieberson_index(self, d):
        """Measure Lieberson's Aw diversity within a population. Aw receives a
        set of variables V with p categories and uses the proportions Yk in
        each category to measure the diversity of the set.
        Example:
        d = {
         'a': [.06, .4, .44, .1],
         'b': [.39, .39, .22],
         'c': [.44, .56],
         'd': [.62, .38],
         'f': [.39, .61],
         'e': [.04, .45, .51]
        }

        Paper:
        Sullivan, John L. "Political Correlates of Social, Economic, and
        Religious Diversity in the American States." The Journal of Politics
        35, no. 1 (1973): 70-84. http://www.jstor.org/stable/2129038.

        Args:
            d (:obj:`dict`): d.keys() contains the variables V. d.values()
                contains lists with the proportions of each category for every
                variable.

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

    print('LOCAL STUDENTS')
    print(indicators.home_study('United Kingdom'))
    print()

    print('LIEBERSON INDEX')
    print(indicators.lieberson_index(
                                     indicators.lieberson_format(
                                        ['gender', 'race'], 'United Kingdom')
                                        ))
    print()


if __name__ == '__main__':
    main()
