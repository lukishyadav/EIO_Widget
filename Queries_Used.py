#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:01:59 2020

@author: lukishyadav
"""

'end_lat','end_long','start_lat','start_long','rental_started_at','rental_booked_at'

Daytona 


select end_lat,end_long,start_lat,start_long,rental_booked_at,rental_started_at,rental_ended_at,vehicle_id,customer_id from daytona_rental_data
where "object_data-is_exempted"=False 
-- and  rental_started_at>'2019-01-10 00:00:01' and rental_started_at<'2020-01-10 00:00:01'



Darwin

select end_lat,end_long,start_lat,start_long,rental_booked_at,rental_started_at,rental_ended_at,vehicle_id,customer_id from darwin_rental_data
where "object_data-is_exempted"=False 
and  rental_started_at>'2020-01-01 00:00:01' 
-- and rental_started_at<'2020-01-10 00:00:01'



Getting data for all tenants for their first 6 months of operation


First date:
    
daytona: 2019-08-12 12:05:36.120 America/Los_Angeles

darwin: 	2017-07-30 18:47:26.866 America/Los_Angeles

darwin e: 2017-07-30 18:47:26.866 America/Los_Angeles

darwin s: 2019-01-03 13:39:35.657 America/Los_Angeles

eiffel: 2018-05-11 20:00:49.644 Europe/Madrid


select * from daytona_rental_data where rental_booked_at>
(select min(rental_booked_at) from daytona_rental_data where "object_data-is_exempted"=False) and rental_booked_at< substr((select min(rental_booked_at) from daytona_rental_data where "object_data-is_exempted"=False),1,20)


Obtaining first 6 months trial query: 
    
select (from_iso8601_date(q.rs) + interval '6' month),q.rs from (select substr(rental_started_at,1,19) as rs, from daytona_rental_data) q