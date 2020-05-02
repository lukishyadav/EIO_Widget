#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:17:11 2020

@author: lukishyadav
"""

select end_lat,end_long,start_lat,start_long,rental_booked_at,rental_started_at,rental_ended_at,vehicle_id,customer_id,"object_data-fare"	
,"object_data-total_to_charge"
"object_data-credit_amount_used" from darwin_east_bay_rental_data
where "object_data-is_exempted"=False 
and  rental_started_at>'2020-03-16 00:00:00' 
-- and rental_started_at<'2020-01-10 00:00:01'




select end_lat,end_long,start_lat,start_long,rental_booked_at,rental_started_at,rental_ended_at,vehicle_id,customer_id,"object_data-fare"	
,"object_data-total_to_charge"
"object_data-credit_amount_used" from darwin_rental_data
where "object_data-is_exempted"=False 
and  rental_started_at>'2020-03-16 00:00:00' 
-- and rental_started_at<'2020-01-10 00:00:01'



select end_lat,end_long,start_lat,start_long,rental_booked_at,rental_started_at,rental_ended_at,vehicle_id,customer_id,"object_data-fare"	
,"object_data-total_to_charge"
"object_data-credit_amount_used" from darwin_sacramento_rental_data
where "object_data-is_exempted"=False 
and  rental_started_at>'2020-03-16 00:00:00' 
-- and rental_started_at<'2020-01-10 00:00:01'