import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def build_data_frame():

    listings = pd.read_csv('data/london_listings.csv')

    df_clean = listings[['id', 'price', 'host_is_superhost', 'latitude', 'longitude', 
                         'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text',
                         'bedrooms', 'beds', 'amenities', 'instant_bookable', 'host_has_profile_pic', 
                         'host_identity_verified', 'number_of_reviews']]

    df_clean = df_clean[df_clean['price'] != 0]

    df_clean = df_clean.set_index('id')

    df_clean['price'] = df_clean['price'].str.replace(',', '')
    df_clean['price'] = df_clean['price'].str.replace('$', '')
    df_clean['price'] = df_clean['price'].astype(float)

    df_clean['host_is_superhost'] = (df_clean['host_is_superhost'] == 't').astype(int)
    df_clean['host_is_superhost'] = df_clean['host_is_superhost'].fillna(0)

    df_clean = pd.get_dummies(df_clean, columns=['neighbourhood_cleansed'])

    df_clean = pd.get_dummies(df_clean, columns=['room_type'])

    df_clean['bathrooms_text'].fillna(2, inplace=True)
    clean_bathrooms_text = {"1 bath": 1,
                        "1 shared bath": 1,
                        "2 baths": 2,
                        "1 private bath": 1,
                        "1.5 shared baths": 1.5,
                        "1.5 baths": 1.5,
                        "0 shared baths": 0,
                        "2.5 shared baths": 2.5,
                        "Shared half-bath": 0.5,
                        "4 baths": 4,
                        "3 baths": 3,
                        "0 baths": 0,
                        "3 shared baths": 3,
                        "3.5 baths": 3.5,
                        "Half-bath": 0.5,
                        "5 baths": 5,
                        "4.5 baths": 4.5,
                        "5 shared baths": 5,
                        "3.5 shared baths": 3.5,
                        "Private half-bath": 0.5,
                        "7 baths": 7,
                        "4 shared baths": 4,
                        "6 baths": 6,
                        "6 shared baths": 6,
                        "5.5 baths": 5.5,
                        "10 baths": 10,
                        "8.5 baths": 8.5,
                        "7 shared baths": 7,
                        "4.5 shared baths": 4.5,
                        "6.5 baths": 6.5,
                        "8 shared baths": 8,
                        "17 baths": 17,
                        "11 baths": 11,
                        "7.5 baths": 7.5,
                        "8 baths": 8,
                        "10.5 baths": 10.5,
                        "9 baths": 9,
                        "12 baths": 12,
                        "9 shared baths": 9,
                        "35 baths": 35,
                        "2 shared baths": 2,
                        "2.5 baths": 2.5}
    df_clean['bathrooms_text'] = df_clean['bathrooms_text'].replace(clean_bathrooms_text)

    df_clean.dropna(subset=['bedrooms','beds'], how='all', inplace=True)

    bedrooms_no_nulls = df_clean[df_clean['bedrooms'].notnull()]
    mean_beds_rounded = bedrooms_no_nulls.groupby(by=['beds']).mean().round(0).reset_index()

    fill_dict1 = mean_beds_rounded.set_index('beds')['bedrooms'].to_dict()
    df_clean['bedrooms'] = df_clean['bedrooms'].replace(np.nan, df_clean['beds'].map(fill_dict1))

    beds_no_nulls = df_clean[df_clean['beds'].notnull()]
    mean_bedrooms_rounded = beds_no_nulls.groupby(by=['bedrooms']).mean().round(0).reset_index()

    fill_dict2 = mean_bedrooms_rounded.set_index('bedrooms')['beds'].to_dict()
    df_clean['beds'] = df_clean['beds'].replace(np.nan, df_clean['bedrooms'].map(fill_dict2))

    df_clean_copy = df_clean.copy()
    df_clean_copy['types'] = df_clean_copy['amenities'].apply(lambda x: type(x))
    df_clean_copy['amenities'] = df_clean_copy['amenities'].apply(lambda x: x.strip("][").split(', ') if "[" in x else [x])
    s = df_clean_copy['amenities']
    mlb = MultiLabelBinarizer()
    amenity_dummies = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=df_clean.index)
    amenity_dummies['price'] = df_clean_copy['price']


    df_clean['instant_bookable'] = (df_clean['instant_bookable'] == 't').astype(int)

    df_clean['host_has_profile_pic'] = (df_clean['host_has_profile_pic'] == 't').astype(int)
    df_clean['host_has_profile_pic'] = df_clean['host_has_profile_pic'].fillna(0)

    df_clean['host_identity_verified'] = (df_clean['host_identity_verified'] == 't').astype(int)
    df_clean['host_identity_verified'] = df_clean['host_identity_verified'].fillna(0)

    amenities = amenity_dummies[['"Air conditioning"', '"TV"', '"Dishwasher"', '"Ironing board"', 
                                 '"Security system"', '"Dryer"', '"Chef\'s kitchen"', '"Coffee maker"', 
                                 '"Balcony"', '"Terrace"', '"Espresso machine"', '"Crib"', 
                                 '"Dining area for 8 people"', '"Indoor fireplace"', '"Private entrance"', 
                                 '"Oven"', '"High chair"', '"Shampoo"' ]]

    drop_amenities = df_clean.drop(['amenities'], axis=1)

    result = pd.concat([drop_amenities, amenities], axis=1)

    return result


def gridsearch_scoring(estimator, param_grid, score, X_train, y_train):

    model_gridsearch = GridSearchCV(estimator, param_grid, n_jobs=-1, verbose=True, scoring=score)
    model_gridsearch.fit(X_train, y_train)
    best_estimator = model_gridsearch.best_estimator_
    best_param = model_gridsearch.best_params_
    best_score = model_gridsearch.best_score_
    print("\n Result of gridsearch")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals, in param_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), str(best_param[param]), str(vals)))

    return best_param, best_estimator, best_score


def score_classifier(model, name, X_train, y_train):
    start_time = time.time()
    cv_scores = cross_validate(model, X_train, y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1), return_train_score=False, scoring=["recall", ])


if __name__ == '__main__':

    build_data_frame()

    rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    xgb = XGBClassifier(n_jobs=-1, random_state=1)
    gbc = GradientBoostingClassifier(random_state=1)

    r,p,f = score_classifier(rf, "random forest classifier")
    r,p,f = score_classifier(xgb, "xgboost classifier")
    r,p,f = socre_classifier(gbc, "gradient boost classifier")

    r,p,f = test_classifier(rf, "random forest classifier")
    r,p,f = test_classifier(xgb, "xgboost classifier")
    r,p,f = test_classifier(gbc, "gradient boost classifier")

    estimators = [['rf', rf], ['xgb', xgb], ['gbc', gbc]]
    voting = VotingClassifier(estimators, voiting='soft')
    r,p,f = test_classifier(voting, "bagging ensemble voting classifier")