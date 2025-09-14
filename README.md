# Tree-Based Models for the Prediction of Forest Fires
A code repository for my 2025 Summer Internship project.

I would like to thank the ICCS for providing the opportunity and my supervisor, Dr Robert Rouse, for his guidance throughout the project.

Note that I will use the term **Forest Fire** throughout, but in general, this project applies to vegetation fires as a whole.

## What?
This project was inspired by the papers:
- [A Global Probability of Fire (PoF) Forecast](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107929)
- [Streamflow Prediction using Artificial Neural Networks and Soil Moisture Proxies](https://www.cambridge.org/core/journals/environmental-data-science/article/streamflow-prediction-using-artificial-neural-networks-and-soil-moisture-proxies/0BD1412AC5E8CED23D4564AECD8F2583)

In particular, the aim is to create a Tree-Based model (so as to allow for relatively direct comparison to the first paper) to predict the occurrence of vegetation fires.

### What's special?
The main changes made in the training process as compared to the original paper were:
- Replacing _Soil Moisture_ with antecedent proxies (an idea taken from the second paper)
- Training on a specific region (in this case, I used the Richardson Backcountry area, in North-east Alberta)

An _antecedent proxy variable_ is just a rolling average over some period of time - I used 30, 90 and 180 day rolling averages of temperature and precipitation for this project.

While the choice to train on a small geographical extent was due to time / storage / memory contstraints, it has seemed to create a reasonably good model that can generalise fairly well to other geographical regions - perhaps worth considering as a takeaway from this project.

## Why?

### Why look at Forest Fires?
Forest fire numbers are vastly higher than ever before - (i could probably put some stats here but i'll find them later. suffice to say there's a lot of them and they're on an ever increasing scale of largeness)

### Why model Forest Fires?
Effective models of forest fires allows for more targeted policy decisions (such as fire-prevention / mitigation measures over more specific periods of time) which allows for a better tradeoff between safety and convenience for everybody involved.

### Why Machine Learning?
Very much a question outside of the scope of this project; the main point is that physical models are (and will always continue to be) invaluable for developing a deeper understanding of the mechanics behind the start and spread of fires, but for a quickly preparable, ready-to-use model, Machine Learning (or other data-driven) models will always be an attractive option

### Why is this project important?
Recalling that the important change in this is **replacing Soil Moisture with antecedent proxy variables**, the main idea is that, of all the input variables used, Soil Moisture is by far the most finickity to measure - all of the other input variables (to the best of my knowledge) for our model can be measured from satellite data, whereas Soil Moisture requires local, on-the-ground measurements. Thus, if we can replace the Soil Moisture variable with a set of 'easily measureable' variables that give similar / the same information, we can create a more useful model for forecasting fires.

## How?

### Get Started
Setup a conda environment with the conda packages from (requirements doc) and pip installs from (requirements.txt which will be created at some point) -- this stuff is yet to be done (feel free to clone this repo though and figure out which packages are needed)

Install **conda** requirements from [requirements.txt](requirements.txt) and **pip** requirements from [pip_requirements.txt](pip_requirements.txt) -- this is currently waaaay more stuff than is required, but also it'll take effort to strip this down, and thus is a future thing to do. To be quite honest, I think they are mostly the same, but probably install the conda one first, then the pip one after.

## Contribute
This project was coded and maintained by myself. Any contributions are welcome.
