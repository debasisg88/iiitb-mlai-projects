# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import zomatopy
import json
import pandas as pd
import re
import smtplib
import config_params

email_response_msg = ""

# Class to Validate Location
class ActionValidateLocation(Action):

    def name(self) -> Text:
        return 'action_validate_location'
    
    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            loc = tracker.get_slot('location')
            if str(loc).lower() not in open('./data/cities.txt', 'r').read():
                return [SlotSet("location", None)]
            else:
                return [SlotSet('location', loc)]
        
        except Exception as e:
            dispatcher.utter_message('Issue in validate location: \n' + str(e))

# Class to Validate Cuisine
class ActionValidateCuisine(Action):

    def name(self) -> Text:
        return 'action_validate_cuisine'
    
    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            cuisines = tracker.get_slot('cuisine')
            if str(cuisines).lower() not in [l.lower() for l in config_params.valid_cuisines]:
                return [SlotSet("cuisine", None)]
            else:
                return [SlotSet('cuisine', cuisines)]
        
        except Exception as e:
            dispatcher.utter_message('Issue in validate cuisines: \n' + str(e))

# Class to Validate Budget
class ActionValidateBudget(Action):

    def name(self) -> Text:
        return 'action_validate_budget'
    
    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            avg_cost_for_two = tracker.get_slot('budget')
            if str(avg_cost_for_two).lower() not in [l.lower() for l in config_params.valid_budgets]:
                return [SlotSet("budget", None)]
            else:
                return [SlotSet('budget', avg_cost_for_two)]
        
        except Exception as e:
            dispatcher.utter_message('Issue in validate budget: \n' + str(e))

# Class to search Restaurants
class ActionSearchRestaurants(Action):
    
    def name(self) -> Text:
        return 'action_restaurant'
    
    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            try:
                global email_response_msg
                email_response_msg = ""
                config = config_params.user_key
                zomato = zomatopy.initialize_app(config)
                
                # Get Location Details
                loc = tracker.get_slot('location')
                location_detail = zomato.get_location(loc, 1)
                d1 = json.loads(location_detail)
                lat = d1["location_suggestions"][0]["latitude"]
                lon = d1["location_suggestions"][0]["longitude"]

                # Get Cusine Details
                cuisines_dict ={'chinese': 25, 
                                'mexican': 73, 
                                'italian': 55, 
                                'american': 1, 
                                'south indian': 85, 
                                'north indian': 50}
                cuisine = str(cuisines_dict.get(tracker.get_slot('cuisine')))
                
                # Get Budget Details
                budget = tracker.get_slot('budget')

                name = []
                addr = []
                costfor2 = []
                rating = []
                cuisines = []
                start = 0
                while start <= 100:
                    results = zomato.restaurant_search("",lat,lon,cuisine,limit=20,start=start, sort='rating', order="desc")
                    d = json.loads(results)
                    if d['results_found'] != 0:
                        for restaurant in d['restaurants']:
                            name.append(restaurant['restaurant']['name'])
                            addr.append(restaurant['restaurant']['location']['address'])
                            rating.append(restaurant['restaurant']['user_rating']['aggregate_rating'])
                            costfor2.append(restaurant['restaurant']['average_cost_for_two'])
                            cuisines.append(restaurant['restaurant']['cuisines'])
                        
                    start = start + 20
                
                restaurant_df = pd.DataFrame({'name': name, 'addr': addr, 'costfor2': costfor2, 
                                                'rating': rating, 'cuisine': cuisines})
                restaurant_df['rating'] = restaurant_df['rating'].apply(pd.to_numeric)

                if budget == 'Lesser than 300':
                    restaurant_df1 = restaurant_df[restaurant_df['costfor2'] < 300]
                elif budget == '300 to 700':
                    restaurant_df1 = restaurant_df[(restaurant_df['costfor2'] >= 300) & 
                                                    (restaurant_df['costfor2'] <= 700)]
                elif budget == 'More than 700':
                    restaurant_df1 = restaurant_df[restaurant_df['costfor2'] > 700]
                else:
                    restaurant_df1 = restaurant_df
                
                response = ""
                if len(restaurant_df1.index) == 0:
                    response = ""
                    dispatcher.utter_message("No Results :-(")
                    return [SlotSet('location', None)]
                else:
                    restaurant_df = restaurant_df1.sort_values(by='rating', ascending=False)
                    restaurant_df['info']="Restaurant "+ restaurant_df['name']+ " in "+ restaurant_df['addr']+" has been rated "+restaurant_df['rating'].astype(str)
                    response='\n'.join(restaurant_df['info'].head(5).tolist())
                    top10='\n'.join(restaurant_df['info'].head(10).tolist())
                    email_response_msg = top10
                    dispatcher.utter_message("Here's what I found:")
                    dispatcher.utter_message(response)
                    return [SlotSet('location', loc)]

            except Exception as e:
                dispatcher.utter_message('Issue in searching restaurants: \n' + str(e))

# Class to Check Email Address
class ActionCheckEmail(Action):
    def name(self) -> Text:
        return 'action_check_email_id'
    
    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            try:
                if tracker.get_slot('email'):
                    email = tracker.get_slot('email')
                    if (re.search('[A-z0-9]+@[A-z0-9]+[.][A-z]+[.]?[A-z]*',email)):
                        return [SlotSet("email", email)]
                    else:
                        return [SlotSet("email", None)]
                else:
                    return [SlotSet("email", None)]
            except Exception as e:
                dispatcher.utter_message('Issue in Checking Email: \n' + str(e))

# Class to Send Email
class ActionSendEmail(Action):
	def name(self) -> Text:
		return 'action_send_email'
		
	def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            try:
                global email_response_msg
                loc = tracker.get_slot('location')
                cuisine = tracker.get_slot('cuisine')
                email_response_msg = (config_params.email_template %(loc, email_response_msg))
                subject='Top rated {} restuartants in {}'.format(cuisine, loc)
                server = smtplib.SMTP('smtp.gmail.com:587')
                server.ehlo()
                server.starttls()
                receiver_email = tracker.get_slot('email')
                server.login(config_params.email, config_params.password)
                message = 'Subject: {}\n\n{}'.format(subject, email_response_msg)
                server.sendmail(receiver_email,receiver_email, message)
                dispatcher.utter_message('Email Sent Successfully...')
            except Exception as e:
                dispatcher.utter_message('Issue in sending message \n' + str(e))