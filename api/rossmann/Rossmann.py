import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Rossmann:

    def __init__(self):
        self.home_path = r'C:\Users\Ganso\Documents\Data-Science\repos\Drugstore-Sales-Forecast\Notebooks'
        self.competition_distance_scaler = pickle.load(
            open(self.home_path + '\parameter\competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(
            open(self.home_path + '\parameter\competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + '\parameter\promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open(self.home_path + '\parameter\year_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open(self.home_path + '\parameter\store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df01):
        ## 1.1. Rename Columns

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
                    'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map(snakecase, cols_old))

        # rename
        df01.columns = cols_new

        ## 1.3. Data Types

        df01['date'] = pd.to_datetime(df01['date'])

        ## 1.5. Fillout NA

        # competition_distance
        df01['competition_distance'] = df01['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        # competition_open_since_month
        df01['competition_open_since_month'] = df01.apply(
            lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x[
                'competition_open_since_month'],
            axis=1
        )

        # competition_open_since_year
        df01['competition_open_since_year'] = df01.apply(
            lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x[
                'competition_open_since_year'],
            axis=1
        )

        # promo2_since_week
        df01['promo2_since_week'] = df01.apply(
            lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'],
            axis=1
        )

        # promo2_since_year
        df01['promo2_since_year'] = df01.apply(
            lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'],
            axis=1
        )

        # promo_interval
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df01['promo_interval'].fillna(0, inplace=True)

        df01['month_map'] = df01['date'].dt.month.map(month_map)

        df01['is_promo'] = df01[['promo_interval', 'month_map']].apply(
            lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0,
            axis=1
        )

        ## 1.6. Change Types

        df01['competition_open_since_month'] = df01['competition_open_since_month'].astype(np.int64)
        df01['competition_open_since_year'] = df01['competition_open_since_year'].astype(np.int64)

        df01['promo2_since_week'] = df01['promo2_since_week'].astype(np.int64)
        df01['promo2_since_year'] = df01['promo2_since_year'].astype(np.int64)

        return df01

    def feature_engineering(self, df02):
        # year
        df02['year'] = df02['date'].dt.year

        # month
        df02['month'] = df02['date'].dt.month

        # day
        df02['day'] = df02['date'].dt.day

        # week of year
        df02['week_of_year'] = df02['date'].dt.isocalendar().week.astype(np.int64)

        # year week
        df02['year_week'] = df02['date'].dt.strftime('%Y-%W')

        # competition since
        df02['competition_since'] = df02.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                           month=x['competition_open_since_month'],
                                                                           day=1), axis=1)

        df02['competition_time_month'] = ((df02['date'] - df02['competition_since']) / 30).apply(
            lambda x: x.days).astype(np.int64)

        # promo since
        df02['promo_since'] = df02['promo2_since_year'].astype(str) + '-' + df02['promo2_since_week'].astype(str)
        df02['promo_since'] = df02['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df02['promo_time_week'] = ((df02['date'] - df02['promo_since']) / 7).apply(lambda x: x.days).astype(np.int64)

        # assortment
        df02['assortment'] = df02['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state holiday
        df02['state_holiday'] = df02['state_holiday'].apply(lambda
                                                                x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # 3.0. FILTRAGEM DE VARIÁVEIS
        ## 3.1. Filtragem das Linhas
        df02 = df02[df02['open'] != '0']

        ## 3.2. Seleção das Colunas
        cols_drop = ['open', 'promo_interval', 'month_map']

        df02 = df02.drop(cols_drop, axis=1)

        return df02

    def data_preparation(self, df05):
        # competition_distance
        df05['competition_distance'] = self.competition_distance_scaler.fit_transform(
            df05[['competition_distance']].values)

        # competition time month
        df05['competition_time_month'] = self.competition_time_month_scaler.fit_transform(
            df05[['competition_time_month']].values)

        # promo time week
        df05['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df05[['promo_time_week']].values)

        # year
        df05['year'] = self.year_scaler.fit_transform(df05[['year']].values)

        # state_holiday - One Hot Encoding
        df05 = pd.get_dummies(df05, prefix=['state_holiday'], columns=['state_holiday'], dtype=np.int64)

        # store_type - Label Encoding
        df05['store_type'] = self.store_type_scaler.fit_transform(df05['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df05['assortment'] = df05['assortment'].map(assortment_dict)

        ### 5.3.2. Response Variable Transformation

        df05['sales'] = np.log1p(df05['sales'])

        ### 5.3.3. Nature Transformation

        # day of week
        df05['day_of_week_sin'] = df05['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df05['day_of_week_cos'] = df05['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))

        # month
        df05['month_sin'] = df05['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df05['month_cos'] = df05['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))

        # day
        df05['day_sin'] = df05['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df05['day_cos'] = df05['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))

        # week of year
        df05['week_of_year_sin'] = df05['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi / 52)))
        df05['week_of_year_cos'] = df05['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi / 52)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month', 'competition_open_since_year', 'promo2',
                         'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                         'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin',
                         'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df05[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # join pred into the original data
        original_data['predictions'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')

