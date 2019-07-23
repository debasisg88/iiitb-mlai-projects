## Story 1
* greet
    - utter_greet
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_chat_restart
    - action_restart

## Story 2
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location":"bangalore"}
    - slot{"location":"bangalore"}
    - action_validate_location
    - slot{"location":"bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 3
* greet
    - utter_greet
* restaurant_search{"location":"kolkata"}
    - slot{"location":"kolkata"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 4
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location":"bangalore"}
    - slot{"location":"bangalore"}
    - action_validate_location
    - slot{"location":"bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 5
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email": "debasis.garabadu@gmail.com"}
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_send_email
    - slot{"email": "debasis.garabadu@gmail.com"}
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 6
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 7
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 8
* greet
    - utter_greet
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email": "debasis.garabadu@gmail.com"}
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_send_email
    - slot{"email": "debasis.garabadu@gmail.com"}
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 9
* greet
    - utter_greet
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 10
* greet
    - utter_greet
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 11
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine": "American"}
    - utter_ask_budget
* restaurant_search{"budget": "Less than 300"}
    - slot{"budget": "Less than 300"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": null}
    - utter_no_restaurants
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 12
* greet
    - utter_greet
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine": "American"}
    - utter_ask_budget
* restaurant_search{"budget": "Less than 300"}
    - slot{"budget": "Less than 300"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": null}
    - utter_no_restaurants
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 13
* greet
    - utter_greet
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - action_validate_cuisine
    - slot{"cuisine":"Mexican"}
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 14
* greet
    - utter_greet
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - utter_ask_location
* restaurant_search{"location":"kolkata"}
    - slot{"location":"kolkata"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 15
* greet
    - utter_greet
* restaurant_search{"cuisine":"Desserts"}
    - slot{"cuisine":"Desserts"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 16
* greet
    - utter_greet
* restaurant_search{"cuisine":"Desserts"}
    - slot{"cuisine":"Desserts"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 17
* greet
    - utter_greet
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - action_validate_cuisine
    - slot{"cuisine": "Chinese"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email": "debasis.garabadu@gmail.com"}
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_send_email
    - slot{"email": "debasis.garabadu@gmail.com"}
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 18
* greet
    - utter_greet
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - action_validate_cuisine
    - slot{"cuisine": "Chinese"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 19
* greet
    - utter_greet
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - action_validate_cuisine
    - slot{"cuisine": "Chinese"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 20
* greet
    - utter_greet
* restaurant_search{"cuisine":"Desserts"}
    - slot{"cuisine":"Desserts"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 21
* greet
    - utter_greet
* restaurant_search{"cuisine":"Desserts"}
    - slot{"cuisine":"Desserts"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 22
* greet
    - utter_greet
* restaurant_search{"cuisine":"Desserts"}
    - slot{"cuisine":"Desserts"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 23
* greet
    - utter_greet
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 24
* greet
    - utter_greet
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 25
* greet
    - utter_greet
* restaurant_search{"budget":"200 to 900"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 26
* greet
    - utter_greet
* restaurant_search{"budget":"200 to 900"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location":"bhubaneswar"}
    - slot{"location":"bhubaneswar"}
    - action_validate_location
    - slot{"location":"bhubaneswar"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 27
* greet
    - utter_greet
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email": "debasis.garabadu@gmail.com"}
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email": "debasis.garabadu@gmail.com"}
    - action_send_email
    - slot{"email": "debasis.garabadu@gmail.com"}
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 28
* greet
    - utter_greet
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bhubaneswar"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 29
* greet
    - utter_greet
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 30
* greet
    - utter_greet
* restaurant_search{"budget":"200 to 900"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 31
* greet
    - utter_greet
* restaurant_search{"budget":"200 to 900"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Chinese"}
    - slot{"cuisine": "Chinese"}
    - utter_ask_budget
* restaurant_search{"budget": "300 to 700"}
    - slot{"budget": "300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location": "bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 32
* greet
    - utter_greet
* restaurant_search{"budget":"200 to 900"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_help
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 33
* greet
    - utter_greet
* restaurant_search{"location":"kolkata", "cuisine":"Italian"}
    - slot{"location":"kolkata"}
    - slot{"cuisine":"Italian"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - action_validate_cuisine
    - slot{"cuisine":"Italian"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 34
* greet
    - utter_greet
* restaurant_search{"location":"kolkata", "cuisine":"Italian"}
    - slot{"location":"kolkata"}
    - slot{"cuisine":"Italian"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - action_validate_cuisine
    - slot{"cuisine":"Italian"}
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 35
* greet
    - utter_greet
* restaurant_search{"location":"kolkata", "cuisine":"bengali"}
    - slot{"location":"kolkata"}
    - slot{"cuisine":"bengali"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 36
* greet
    - utter_greet
* restaurant_search{"location":"kolkata", "cuisine":"bengali"}
    - slot{"location":"kolkata"}
    - slot{"cuisine":"bengali"}
    - action_validate_location
    - slot{"location":"kolkata"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kolkata"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 37
* greet
    - utter_greet
* restaurant_search{"location":"deoghar", "cuisine":"Mexican"}
    - slot{"location":"deoghar"}
    - slot{"cuisine":"Mexican"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bangalore"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 38
* greet
    - utter_greet
* restaurant_search{"location":"deoghar", "cuisine":"Mexican"}
    - slot{"location":"deoghar"}
    - slot{"cuisine":"Mexican"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "bangalore"}
    - slot{"location": "bangalore"}
    - action_validate_location
    - slot{"location": "bangalore"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"bangalore"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 39
* greet
    - utter_greet
* restaurant_search{"location":"deoghar", "cuisine":"Mexican"}
    - slot{"location":"deoghar"}
    - slot{"cuisine":"Mexican"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 40
* greet
    - utter_greet
* restaurant_search{"location":"deoghar", "cuisine":"Mexican"}
    - slot{"location":"deoghar"}
    - slot{"cuisine":"Mexican"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "buxar"}
    - slot{"location": "buxar"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "chennai"}
    - slot{"location": "chennai"}
    - action_validate_location
    - slot{"location": "chennai"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"chennai"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 41
* greet
    - utter_greet
* restaurant_search{"location":"deoghar", "cuisine":"Mexican"}
    - slot{"location":"deoghar"}
    - slot{"cuisine":"Mexican"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "buxar"}
    - slot{"location": "buxar"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "chennai"}
    - slot{"location": "chennai"}
    - action_validate_location
    - slot{"location": "chennai"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"chennai"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 42
* greet
    - utter_greet
* restaurant_search{"location":"udupi", "cuisine":"odia"}
    - slot{"location":"udupi"}
    - slot{"cuisine":"odia"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "cuttack"}
    - slot{"location": "cuttack"}
    - action_validate_location
    - slot{"location": "cuttack"}
    - slot{"cuisine":"odia"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"cuttack"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 43
* greet
    - utter_greet
* restaurant_search{"location":"udupi", "cuisine":"odia"}
    - slot{"location":"udupi"}
    - slot{"cuisine":"odia"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "cuttack"}
    - slot{"location": "cuttack"}
    - action_validate_location
    - slot{"location": "cuttack"}
    - slot{"cuisine":"odia"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"cuttack"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 44
* greet
    - utter_greet
* restaurant_search{"location":"udupi", "cuisine":"odia"}
    - slot{"location":"udupi"}
    - slot{"cuisine":"odia"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "buxar"}
    - slot{"location": "buxar"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 45
* greet
    - utter_greet
* restaurant_search{"location":"udupi", "cuisine":"odia"}
    - slot{"location":"udupi"}
    - slot{"cuisine":"odia"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "cuttack"}
    - slot{"location": "cuttack"}
    - action_validate_location
    - slot{"location": "cuttack"}
    - slot{"cuisine":"odia"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"cuttack"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 46
* greet
    - utter_greet
* restaurant_search{"location":"udupi", "cuisine":"odia"}
    - slot{"location":"udupi"}
    - slot{"cuisine":"odia"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "panipat"}
    - slot{"location": "panipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "cuttack"}
    - slot{"location": "cuttack"}
    - action_validate_location
    - slot{"location": "cuttack"}
    - slot{"cuisine":"odia"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - utter_ask_budget
* restaurant_search{"budget":"More than 700"}
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"cuttack"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 47
* greet
    - utter_greet
* restaurant_search{"location":"delhi", "budget":"Lesser than 300"}
    - slot{"location":"delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location":"delhi"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 48
* greet
    - utter_greet
* restaurant_search{"location":"delhi", "budget":"Lesser than 300"}
    - slot{"location":"delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location":"delhi"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 49
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 50
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 51
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 52
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "arrah"}
    - slot{"location": "arrah"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 53
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "arrah"}
    - slot{"location": "arrah"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_budget
    - slot{"budget":"Lesser than 300"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 54
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"Lesser than 300"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"Lesser than 300"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "arrah"}
    - slot{"location": "arrah"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 55
* greet
    - utter_greet
* restaurant_search{"location":"delhi", "budget":"200 to 900"}
    - slot{"location":"delhi"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"delhi"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 56
* greet
    - utter_greet
* restaurant_search{"location":"delhi", "budget":"200 to 900"}
    - slot{"location":"delhi"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"delhi"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine":"Mexican"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 57
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"200 to 900"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "arrah"}
    - slot{"location": "arrah"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 58
* greet
    - utter_greet
* restaurant_search{"location":"bidhannagar", "budget":"200 to 900"}
    - slot{"location":"bidhannagar"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "arrah"}
    - slot{"location": "arrah"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine":"American"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 59
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"300 to 700"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location": "kanpur"}
    - action_validate_location
    - slot{"location": "kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 60
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"300 to 700"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location": "kanpur"}
    - action_validate_location
    - slot{"location": "kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 61
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"300 to 700"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location":"kanpur"}
    - action_validate_location
    - slot{"location":"kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 62
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"300 to 700"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location":"kanpur"}
    - action_validate_location
    - slot{"location":"kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 63
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"300 to 700"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 64
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"300 to 700"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 65
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"300 to 700"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 66
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"300 to 700"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 67
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"300 to 700"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 68
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"300 to 700"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"300 to 700"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 69
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"200 to 900"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location":"delhi"}
    - action_validate_location
    - slot{"location":"delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 70
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"200 to 900"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location":"delhi"}
    - action_validate_location
    - slot{"location":"delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 71
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"200 to 900"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location":"kanpur"}
    - action_validate_location
    - slot{"location":"kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 72
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"200 to 900"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "kanpur"}
    - slot{"location":"kanpur"}
    - action_validate_location
    - slot{"location":"kanpur"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"kanpur"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 73
* greet
    - utter_greet
* restaurant_search{"cuisine":"North Indian", "budget":"200 to 900"}
    - slot{"cuisine": "North Indian"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": "North Indian"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 74
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"200 to 900"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 75
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"200 to 900"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 76
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"200 to 900"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 77
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"200 to 900"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_validate_location
    - slot{"location": "delhi"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"delhi"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 78
* greet
    - utter_greet
* restaurant_search{"cuisine":"bihari", "budget":"200 to 900"}
    - slot{"cuisine":"bihari"}
    - slot{"budget":"200 to 900"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Mexican"}
    - slot{"cuisine": "Mexican"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_ask_location
* restaurant_search{"location": "tenali"}
    - slot{"location": "tenali"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 79
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"Chinese", "budget":"More than 700"}
    - slot{"location":"goa"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"More than 700"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - action_validate_budget
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 80
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"Chinese", "budget":"More than 700"}
    - slot{"location":"goa"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"More than 700"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - action_validate_budget
    - slot{"budget":"More than 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 81
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"goa"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 82
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"goa"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 83
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"goa"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 84
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"goa"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 85
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"goa"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 86
* greet
    - utter_greet
* restaurant_search{"location":"goa", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"goa"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location":"goa"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 87
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 88
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 89
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 90
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 91
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 92
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 93
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 94
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 95
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 96
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"Chinese", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"Chinese"}
    - action_validate_cuisine
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 97
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 98
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 99
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 100
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"300 to 700"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"300 to 700"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"300 to 700"}
    - action_validate_budget
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 101
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 102
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 103
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* affirm
    - utter_ask_email_id
* ask_email_address{"email":"debasis.garabadu@gmail.com"}
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_check_email_id
    - slot{"email":"debasis.garabadu@gmail.com"}
    - action_send_email
    - utter_goodbye
    - utter_chat_restart
    - action_restart

## Story 104
* greet
    - utter_greet
* restaurant_search{"location":"sultanpur", "cuisine":"turkish", "budget":"200 to 900"}
    - slot{"location":"sultanpur"}
    - slot{"cuisine":"turkish"}
    - slot{"budget":"200 to 900"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "sonipat"}
    - slot{"location": "sonipat"}
    - action_validate_location
    - slot{"location": null}
    - utter_location_not_operated
* affirm
    - utter_ask_location
* restaurant_search{"location": "goa"}
    - slot{"location": "goa"}
    - action_validate_location
    - slot{"location": "goa"}
    - slot{"cuisine":"turkish"}
    - action_validate_cuisine
    - slot{"cuisine": null}
    - utter_cuisine_not_valid_aswell
    - utter_ask_cuisine
* restaurant_search{"cuisine":"Chinese"}
    - slot{"cuisine":"Chinese"}
    - slot{"budget":"200 to 900"}
    - action_validate_budget
    - slot{"budget": null}
    - utter_budget_not_valid_aswell
    - utter_ask_budget
* restaurant_search{"budget":"300 to 700"}
    - slot{"budget":"300 to 700"}
    - utter_looking_for_restaurants
    - action_restaurant
    - slot{"location":"goa"}
    - utter_ask_send_to_email
* deny
    - utter_goodbye
    - utter_chat_restart
    - action_restart