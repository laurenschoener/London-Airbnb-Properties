import pandas as pd
import numpy as np

def build_data_frame():

    listings = pd.read_csv('data/london_listings.csv')

    df_clean = listings[['id', 'price', 'host_is_superhost', 'neighbourhood_cleansed', 
                         'property_type', 'room_type', 'accommodates', 'bathrooms_text', 
                         'bedrooms', 'beds', 'instant_bookable', 'host_has_profile_pic', 
                         'host_identity_verified', 'number_of_reviews']]

    df_clean = df_clean.set_index('id')

    df_clean['price'] = df_clean['price'].str.replace(',', '')
    df_clean['price'] = df_clean['price'].str.replace('$', '')
    df_clean['price'] = df_clean['price'].astype(float)

    df_clean['host_is_superhost'] = (df_clean['host_is_superhost'] == 't').astype(int)
    df_clean['host_is_superhost'] = df_clean['host_is_superhost'].fillna(0)

    clean_neighbourhoods = {"Barking and Dagenham": 1,
                        "Barnet": 2,
                        "Bexley": 3,
                        "Brent": 4,
                        "Bromley": 5,
                        "Camden": 6,
                        "City of London": 7,
                        "Croydon": 8,
                        "Ealing": 9,
                        "Enfield": 10,
                        "Greenwich": 11,
                        "Hackney": 12,
                        "Hammersmith and Fulham": 13,
                        "Haringey": 14,
                        "Harrow": 15,
                        "Havering": 16,
                        "Hillingdon": 17,
                        "Hounslow": 18,
                        "Islington": 19,
                        "Kensington and Chelsea": 20,
                        "Kingston upon Thames": 21,
                        "Lambeth": 22,
                        "Lewisham": 23,
                        "Merton": 24,
                        "Newham": 25,
                        "Redbridge": 26,
                        "Richmond upon Thames": 27,
                        "Southwark": 28,
                        "Sutton": 29,
                        "Tower Hamlets": 30,
                        "Waltham Forest": 31,
                        "Wandsworth": 32,
                        "Westminster": 33}

    df_clean['neighbourhood_cleansed'] = df_clean['neighbourhood_cleansed'].replace(clean_neighbourhoods)


    clean_property_type = {"Entire apartment": 1,
                       "Private room in apartment": 2,
                       "Private room in bed and breakfast": 3,
                       "Private room in house": 4,
                       "Entire townhouse": 5,
                       "Private room in townhouse": 6,
                       "Entire condominium": 7,
                       "Entire serviced apartment": 8,
                       "Room in aparthotel": 9,
                       "Room in serviced apartment": 10, 
                       "Entire house": 11,
                       "Private room in loft": 12,
                       "Private room": 13,
                       "Shared room in apartment": 14,
                       "Tiny house": 15, 
                       "Entire guest suite": 16,
                       "Private room in condominium": 17,
                       "Entire loft": 18,
                       "Houseboat": 19,
                       "Private room in bungalow": 20,
                       "Private room in cottage": 21,
                       "Entire guesthouse": 22, 
                       "Private room in guesthouse": 23,
                       "Shared room in house": 24,
                       "Entire cabin": 25,
                       "Room in bed and breakfast": 26,
                       "Private room in yurt": 27, 
                       "Private room in serviced apartment": 28, 
                       "Boat": 29, 
                       "Private room in guest suite": 30,
                       "Entire cottage": 31, 
                       "Private room in parking space": 32, 
                       "Entire place": 33,
                       "Private room in villa": 34, 
                       "Private room in boat": 35, 
                       "Shared room in loft": 36,
                       "Entire bungalow": 37,
                       "Camper/RV": 38,
                       "Yurt": 39, 
                       "Shared room in bed and breakfast": 40, 
                       "Shared room": 41, 
                       "Shared room in condominium": 42, 
                       "Room in boutique hotel": 43, 
                       "Shared room in guest suite": 44, 
                       "Shared room in hostel": 45, 
                       "Entire home/apt": 46,
                       "Private room in hut": 47,
                       "Entire villa": 48,
                       "Private room in chalet": 49,
                       "Shared room in bungalow": 50,
                       "Private room in camper/rv": 51,
                       "Shared room in townhouse": 52,
                       "Private room in tiny house": 53, 
                       "Private room in cabin": 54,
                       "Entire floor": 55, 
                       "Entire chalet": 56,
                       "Earth house": 57,
                       "Room in hostel": 58,
                       "Private room in hostel": 59,
                       "Shared room in serviced apartment": 60, 
                       "Private room in floor": 61,
                       "Private room in lighthouse": 62,
                       "Barn": 63,
                       "Shared room in villa": 64,
                       "Room in hotel": 65, 
                       "Private room in island": 66, 
                       "Private room in earth house": 67,
                       "Private room in treehouse": 68,
                       "Shared room in farm stay": 69,
                       "Shared room in bus": 70,
                       "Shared room in guesthouse": 71,
                       "Private room in houseboat": 72, 
                       "Castle": 73,
                       "Campsite": 74, 
                       "Private room in farm stay": 75, 
                       "Private room in casa particular": 76,
                       "Shared room in hotel": 77,
                       "Room in apartment": 78,
                       "Private room in dome house": 79,
                       "Casa particular": 80,
                       "Shared room in boutique hotel": 81,
                       "Hut": 82,
                       "Private room in tent": 83,
                       "Room in minsu": 84,
                       "Island": 85,
                       "Private room in bus": 86,
                       "Shared room in earth house": 87,
                       "Private room in shepherd's hut": 88,
                       "Shared room in tent": 89,
                       "Private room in castle": 90,
                       "Lighthouse": 91,
                       "Dome house":92}

    df_clean['property_type'] = df_clean['property_type'].replace(clean_property_type)

    clean_room_type = {"Entire home/apt": 1,
                   "Private room": 2,
                   "Hotel room": 3,
                   "Shared room": 4}

    df_clean['room_type'] = df_clean['room_type'].replace(clean_room_type)

    df_clean['bathrooms_text'].fillna(2, inplace=True)

    clean_bathrooms_text = {"1 bath": 1,
                        "1 shared bath": 2,
                        "2 baths": 3,
                        "1 private bath": 4,
                        "1.5 shared baths": 5,
                        "1.5 baths": 6,
                        "0 shared baths": 7,
                        "2.5 shared baths": 8,
                        "Shared half-bath": 9,
                        "4 baths": 10,
                        "3 baths": 11,
                        "0 baths": 12,
                        "3 shared baths": 13,
                        "3.5 baths": 14,
                        "Half-bath": 15,
                        "5 baths": 16,
                        "4.5 baths": 17,
                        "5 shared baths": 18,
                        "3.5 shared baths": 19,
                        "Private half-bath": 20,
                        "7 baths": 21,
                        "4 shared baths": 22,
                        "6 baths": 23,
                        "6 shared baths": 24,
                        "5.5 baths": 25,
                        "10 baths": 26,
                        "8.5 baths": 27,
                        "7 shared baths": 28,
                        "4.5 shared baths": 29,
                        "6.5 baths": 30,
                        "8 shared baths": 31,
                        "17 baths": 32,
                        "11 baths": 33,
                        "7.5 baths": 34,
                        "8 baths": 35,
                        "10.5 baths": 36,
                        "9 baths": 37,
                        "12 baths": 38,
                        "9 shared baths": 39,
                        "35 baths": 40,
                        "2 shared baths": 41,
                        "2.5 baths": 42}

    df_clean['bathrooms_text'] = df_clean['bathrooms_text'].replace(clean_bathrooms_text)

    df_clean.dropna(subset=['bedrooms','beds'], how='all', inplace=True)

    bedrooms_no_nulls = df_clean[df_clean['bedrooms'].notnull()]
    mean_beds_rounded = bedrooms_no_nulls.groupby(by=['beds']).mean().round(0).reset_index()

    fill_dict1 = mean_beds_rounded.set_index('beds')['bedrooms'].to_dict()
    df_clean['bedrooms'] = df_clean['bedrooms'].replace(np.nan, df_clean['beds'].map(fill_dict1))

    beds_no_nulls = df_clean[df_clean['beds'].notnull()]
    mean_bedrooms_rounded = bedrooms_no_nulls.groupby(by=['bedrooms']).mean().round(0).reset_index()

    fill_dict2 = mean_beds_rounded.set_index('bedrooms')['beds'].to_dict()
    df_clean['beds'] = df_clean['beds'].replace(np.nan, df_clean['bedrooms'].map(fill_dict2))

    df_clean['instant_bookable'] = (df_clean['instant_bookable'] == 't').astype(int)

    df_clean['host_has_profile_pic'] = (df_clean['host_has_profile_pic'] == 't').astype(int)
    df_clean['host_has_profile_pic'] = df_clean['host_has_profile_pic'].fillna(0)

    df_clean['host_identity_verified'] = (df_clean['host_identity_verified'] == 't').astype(int)
    df_clean['host_identity_verified'] = df_clean['host_identity_verified'].fillna(0)

    df_clean['beds'].fillna(7, inplace=True)

    return df_clean


if __name__ == '__main__':

    build_data_frame()