import sys
import pandas as pd


class Indicators():

    def __init__(self, data):
        self.data = data

    def people_diversity(self, *args):
        df = self.data[self.data.is_current == 1].drop_duplicates('person_id')
        city_gender_pop = df.groupby(list(args)).count()['person_id']
        city_pop = df.groupby(args[0]).count()['person_id']
        idx = city_pop.where(city_pop > 25).dropna() \
                      .sort_values(ascending=False).index
        div = city_gender_pop / city_pop
        return div.reindex(idx, level=0)


def main():
    data = pd.read_csv(sys.argv[1])
    indicators = Indicators(data)

    gender_diversity = indicators.people_diversity('city', 'gender')
    print(gender_diversity)

    degree_diversity = indicators.people_diversity('degree_type', 'gender')
    print(degree_diversity)
    print(indicators.people_diversity('gender', 'degree_type') > 0.2)


if __name__ == '__main__':
    main()
