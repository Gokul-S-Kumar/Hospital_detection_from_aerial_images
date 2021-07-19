# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

# Script for downloading hospital addresses by cities in Maharashtra using Google Places API from GCP. 
import pandas as pd 
import numpy as np 
import requests
import urllib.parse
import json
import time
import string


if __name__ == '__main__':
    
    # Reading dataframe containing the names of cities in Maharashtra
    df_cities = pd.read_csv('../input/maharashtra_cities.csv')
    # Getting the list of cities
    cities = list(df_cities['City'])
    for city in cities:
    
        api_key = '' # Insert your Google Maps API key here
        
        parameters = {'query' : 'hospitals in {}'.format(city), 'key' : api_key} # Querying hospitals in each city

        url = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
        
        final_data = []
        
        while True:
        
            response = requests.get(url, params = parameters)
            
            jj = json.loads(response.text)
            results = jj['results']
            
            #Extracting necessary fields from the response
            for result in results:
                name = result['name']
                address = result['formatted_address']
                place_id = result['place_id']
                lat = result['geometry']['location']['lat'] # Extracting latitude
                lng = result['geometry']['location']['lng'] # Extracting longitude
                types = result['types']
                
                data = [name, address, place_id, lat, lng, types]
                final_data.append(data)
            
            time.sleep(5) # Time-gap for getting the next_page_token for getting max results of each city, i.e., 60.
  
            if 'next_page_token' not in jj:
                break
            else:
                next_page_token = jj['next_page_token']
            
            parameters['pagetoken'] = next_page_token

        labels = ['place_name', 'address', 'place_id', 'latitude', 'longitude', 'types']

        df = pd.DataFrame.from_records(final_data, columns = labels)
        df.to_csv('./addresses/{}.csv'.format(city))