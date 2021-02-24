# court_volume

An Analysis of Court Volume

## About

This project analyzes data about Court operations in Cook County during the COVID-19 Pandemic.

Central Question to be answered: How many cases would have been disposed during the pandemic that would have been resolved in a normal year?

Motivation: Identify the potential impact of diposing cases by estimating the court backlog. 

## Core Concepts

1. Estimate how many cases "would have been" by forecasting the volume of disposed cases into the pandemic months. 

2. Train a forecasting model (Facebook Prohpet) on court and arrest volume data from 2014 to 2018, then predict next in sequence. 

3. Calculate the difference between actual disposed cases and forecased cases of the same type. 

4. Definition: A disposed case in this analysis is one of Dismissal, Guilty Plea, Not Guilty, or Guilty at Trial. 

## Data Source

Cook County Open Data Portal

## Data Visuals

MongoDB Link (TBP)
