import sys
import pandas as pd


class Indicators():

    def __init__(self, data):
        self.data = data
        self.groupsum = lambda x: x / x.sum()

    def degree_diversity(self, *args, city_level=False, country=None,
                         thresh=25):
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
        df = self.data.dropna(subset=['person_id'])
        df = df[(df.is_current == 1) & (df.primary_role == 'company')]
        df = df[df.degree_type.isin(['MBA', 'PhD', 'Postgraduate',
                                     'Undergraduate'])]
        if city_level:
            df = df[df.country == country]
            idx = self.reindexing(thresh, location='city', country=country)
            return df.groupby(list(args))['person_id'] \
                     .count().groupby(level=[args[0], args[-1]]) \
                     .transform(self.groupsum).reindex(idx, level=0) * 100
        else:
            idx = self.reindexing(thresh, location='country')
            return df.groupby(list(args))['person_id'] \
                     .count().groupby(level=[args[0], args[-1]]) \
                     .transform(self.groupsum).reindex(idx, level=0) * 100

    def people_diversity(self, *args, thresh=25):
        """Find the gender / ethnic diversity of the people that are currently
        working in an area.

        Args:
            thresh (:obj:`int`): Filter out instances with a count lower than
                the threshold.
            *args: Usually location (city, country, continent), gender or
                ethnicity. Note that the first argument is used to normalise
                the data.

        Return:
            div (:obj:`pandas.DataFrame`): A DataFrame grouped by the args
                and reindexed based on the number of women/men in the cities.

        """
        df = self.data[(self.data.is_current == 1)
                       & (self.data.primary_role == 'company')] \
                 .drop_duplicates('person_id')
        nominator = df.groupby(list(args)).count()['person_id']
        denominator = df.groupby(args[0]).count()['person_id']
        if args[0] == 'country':
            idx = self.reindexing(thresh, location='country')
        else:
            idx = self.reindexing(thresh, location='city')
        return (nominator / denominator).reindex(idx, level=0) * 100

    def reindexing(self, thresh, location='country', country=None):
        """"""
        df = self.data[(self.data.is_current == 1)
                       & (self.data.primary_role == 'company')] \
                 .drop_duplicates('person_id')
        if country:
            df = df[df.country == country]
        grouped = df.groupby(location).count()['person_id']
        return grouped.where(grouped > thresh) \
                      .dropna().sort_values(ascending=False).index

    def city_role_company(self, *args):
        """Count the number of women/men in each job type and every company
            size.

        Args:
            *args: Usually location (city, country, continent), gender or
                ethnicity, company size and job type.

        Return:
            (:obj:`pandas.DataFrame`): A DataFrame grouped by the args.

        """
        df = self.data.dropna(subset=['person_id'])
        df = df[(df.is_current == 1) & (df.primary_role == 'company')]
        return df.groupby(list(args)).count()['person_id']
        # return self.data[self.data.primary_role == 'company'] \
        #            .drop_duplicates('person_id') \
        #            .groupby(list(args)) \
        #            .count()['person_id']

    def home_study(self, country):
        df = self.data[self.data.country == country]
        local_uni = df[df.institution_id
                         .isin(set(df.org_id))].person_id \
                                               .unique().shape[0]

        all_universities = df.person_id.unique().shape[0]
        return (local_uni / all_universities) * 100

    def lieberson_format(self, cols, country_level=False, city_level=False,
                         country=None):
        """Format data for Lieberson index."""
        df = self.data[self.data.primary_role == 'company'] \
                 .drop_duplicates('person_id')
        if country_level:
            dfs = [df[df.country == country]
                   for country in df.country.unique()]
            country_level_format = {}
            for df in dfs:
                d = {}
                for col in cols:
                    d[col] = list(df[col].value_counts() / df.shape[0])
                country_level_format[df['country'].unique()[0]] = d
            return country_level_format

        if city_level and country is not None:
            dfs = [df[df.city == city]
                   for city in df[df.country == country].city.unique()]
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
    ind = Indicators(data)

    # 1. Gender diversity (city level)
    country_gender = ind.people_diversity('country', 'gender')

    # 2. Ethnic diversity (city level)
    country_ethnicity = ind.people_diversity('country', 'race')

    # 3. Gender distribution for degrees
    degree_gender = ind.degree_diversity('country', 'degree_type', 'gender')

    # 4. Ethnic distribution for degrees
    degree_ethnicity = ind.degree_diversity('country', 'degree_type', 'race')

    # 5. Gender distribution for degrees - city level
    degree_gender = ind.degree_diversity('country', 'degree_type', 'gender',
                                         city_level=True, country='Germany')

    # 6. Ethnic distribution for degrees - city level
    degree_ethnicity = ind.degree_diversity('city', 'degree_type', 'race',
                                            city_level=True, country='Germany')

    # 7. Gender diversity in roles (city level)
    role_comp_gender = ind.city_role_company('country', 'job_type', 'gender')

    # 8. Ethnic diversity in roles (city level)
    role_comp_ethnicity = ind.city_role_company('country', 'job_type', 'race')

    # 9. Gender diversity in categories (city level)
    cat_comp_gender = ind.city_role_company('country', 'category_group_list',
                                            'gender')

    # 10. Ethnic diversity in categories (city level)
    cat_comp_ethnicity = ind.city_role_company('country',
                                               'category_group_list', 'race')

    # 11. Gender diversity in categories
    cat_gender = ind.city_role_company('category_group_list', 'gender')

    # 12. Ethnic diversity in categories
    cat_ethnicity = ind.city_role_company('category_group_list', 'race')

    # 13. Lieberson index (intersectionality) - city level
    data_formatting = ind.lieberson_format(['gender', 'race'], city_level=True,
                                           country='Germany')
    lieberson_index_cities = {k: ind.lieberson_index(v)
                              for k, v in data_formatting.items()}

    # 14. Lieberson index (intersectionality) - country level
    data_formatting = ind.lieberson_format(['gender', 'race'],
                                           country_level=True)
    lieberson_index_countries = {k: ind.lieberson_index(v)
                                 for k, v in data_formatting.items()}

    # 15. Studied at home vs abroad
    work_and_study_place = {country: ind.home_study(country)
                            for country in data.country.unique()}


if __name__ == '__main__':
    main()
