# Datasets

- An overarching problem with these datasets is that they work with incredibly precise data at an enormous scale. This becomes a large problem because to keep the data up to date, new data has to either be modelled or a lot of resources need to go into gathering the data wherein new biases can be introduced. The HNO data is a great example of this, where the data gathered needs to be approved by the governmental agency of the country. This makes this process very susceptible to pressure as the humanitarian individual collecting said data can be forces into report certain more favourable numbers. On the other side an example is the INFORM severity data, which is created through an aggregate model. This means that not only is this data based on a older time when the country's demographic could have been different but also this modelled data would have the same biases that the original data collected several years ago would have had. 

- A lot of the data collected by hand needs an individual to be physically present on the location, which immediately biases against places which are too dangerous. As such, these places don't get surveyed and data on them isn't available which is unfortunate since these are often the places which need the most help. 

- Additionally a lot of this data is often used to make decisions and ask donators for money. This means that often it's aimed to be the bare minimum since that's what has the best chance for donators to say yes. This means that it often ends up being what crisis absolutely bare minimum NEED, with no room for any improvement on top of that. 

# Part 1

- The big limitation here is that we only considered single country plans with regards to the HRPs. The reason we did this is because we really wanted to focus on each country individually, but it also likely means we underestimated exactly how much money is required since the multi-country plans cost also need to accommadated. 

- Another issue is that we extracted the people in need based simply on the "All" cluster in the HNOs, which means quite a bit of nuance is lost. This was done on purpose because we wanted to provide a large oversight of the data, and we will do an analysis on a more granular level of clusters later in the notebook to make up for this limitation.

# Part 2

- Here when defining the criteria for neglect, we used only OCHA data. This means that there is no external funding taken into account, thus if a crisis was 'neglected' by OCHA but this was actually because it received plenty of funding from other sources, our analysis here does not take that into account. 

- We also used a hard cutoff criteria of the severity being >-3.0 for our invisible crisis analysis. The severity is a continuous variable, and even though the value of 3 is based in past occurences, ultimately as a number on the metric its representing, this threshold is somewhat arbitrary. For the future, it would be better to develop a custom metric for which a threshold is not arbitrary, but for the purposes of this we were satisfied with out decision. 