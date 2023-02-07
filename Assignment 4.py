import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import t
import matplotlib.pyplot as plt
seed=1110
np.random.seed(seed)

#1.1
df = pd.read_csv("data.csv")
#print("The first 5 rows: \n",df.head(5))

#1.2 Use pd to reveal Dtype of features and indicate if any null

#print(df.dtypes)
#print(df.isnull().sum())

#1.3 Get summary statistics of data. Lowest / highest std dev? What was the age of youngest player in the WC?
#Use df.describe() to get a summary of the data.
summary = df.describe()

#Change display options to higher columns, as it is annoying when they get cut off.
pd.options.display.max_columns = 999

print("OK")
print(summary)

#Create a dataframe of the standard deviations, then display the info.
summary_of_standard_deviation_row = (summary.loc[["std"],:])
print(summary_of_standard_deviation_row)
print(f"The maximum and minimum of the standard deviations are: {summary_of_standard_deviation_row.T.max().values} {summary_of_standard_deviation_row.T.min().values} from the Value and Reputation features, respectively.")

#Create another summary dataframe, this time of age, to find the info asked.
summary_of_age_column = (summary.loc[:,["Age"]])
print(f"The summary of the Age column is: {summary_of_age_column} and the max value is: {summary_of_age_column.loc['max',:].values}")

#Question 1.4. Display in one plot and determine if any distributions are gaussian.
df_with_only_numbers = df.select_dtypes(include=[np.number])


"""
fig, axs = plt.subplots(4, 3,figsize=(18,10)) #Fits very nicely on my screen.
axes = axs.ravel()


for i, col in enumerate(df_with_only_numbers.columns):

    df_with_only_numbers[col].hist( ax=axes[i],bins=20, alpha=1, color='blue')
    df_with_only_numbers[col].plot(kind='kde', ax=axes[i], secondary_y=True,color ='red')
    axes[i].set_title(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout(w_pad=0.5,h_pad=1)
#plt.gcf().subplots_adjust(bottom=0.05,top=-0.05)
plt.show()
"""
#The lines which look guassian are height, weight, potential, age, and overall. Weak Foot and Skill Moves look fairly gaussian as well.


#1.5"One Hot encode" aka make dummies.
dfdummies = pd.get_dummies(df) #All that is neccessary is this, as it turns all the catagorical variables into 1's and 0's.
#print(dfdummies)
'''
#1.6Plot Shared element is Value between them, so set x to value.


first_plot = sns.jointplot(x='Value', y='Wage', data=dfdummies, marginal_kws=dict(bins=10))


second_plot = sns.jointplot(x='Value', y='Overall', data=dfdummies, marginal_ticks=True,marginal_kws=dict(bins=10))
#second_plot.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False,)
#plt.show()

'''
#1.7. It is clear from 1.6 that the Value and Wage are not properly distributed. As such, they will undergo transformation.

def log_func(x):
    return np.log(x)
df_transformed_Value = df.Value.apply(log_func)
df_transformed_Wage = df.Wage.apply(log_func)


df_transformed = np.vstack((df_transformed_Value,df_transformed_Wage,df.Overall))

'''
third_plot = sns.jointplot(x=df_transformed[0],y=df_transformed[1])
third_plot.ax_joint.set_xlabel("Log Value")
third_plot.ax_joint.set_ylabel("Log Wage")

fourth_plot = sns.jointplot(x=df_transformed[0], y=df_transformed[2])
fourth_plot.ax_joint.set_xlabel("Log Value")
fourth_plot.ax_joint.set_ylabel("Overall")
plt.show()
'''
#Q1.9
dfdummies['Transformed Value'] = df_transformed[0]
dfdummies['Transformed Wage'] = df_transformed[1]
df_correlation_all = (dfdummies.corr(method='pearson'))
df_correlation_Values = (df_correlation_all.loc[:,['Value','Transformed Value']])


#Q10
print(df_correlation_Values)
#The greatest correlation between Value and another feature is Wage, whereas for the Log. Transformed Value it is Overall.
#The greatest negative correlation for Value is being a Reserve Postion, which is the greatest negative correlation for Transformed Value as well.
#The negative correlations are implying that if a player has these traits, their value to the club is worse.
#Positive correlations imply statistics which, when increased, lead to increase in value of the player to the club.

#Q11

"""
Let's train a model to predict player Value using all features except some (Hint: think about those which you transformed)¶
This time instead of R-squared, use the mean_squared_error to calculate Root Mean Squared Error (RMSE) as your model scorer
Split the data into train and test with test_size=0.2, random_state=seed
Pick LinearRegression() from sklearn as your model
Report both prediction (i.e., on training set) and generalization (i.e., on test set) RMSE scores of your model
"""

altered_data = dfdummies.drop(["Value","Wage"],axis='columns')
y = altered_data["Transformed Value"]
print("hi, eric",y)
X = altered_data.drop('Transformed Value',axis='columns')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

mean_squared_error_model = LinearRegression()
mean_squared_error_model.fit(X_train, y_train)
mse_model_coef = mean_squared_error_model.coef_

mse_model_y_train_predictions = mean_squared_error_model.predict(X_train)
mse_model_y_test_predictions = mean_squared_error_model.predict(X_test)


mean_squared_error_train = mean_squared_error(y_train,mse_model_y_train_predictions)
mean_squared_error_test = mean_squared_error(y_test,mse_model_y_test_predictions)
print(np.sqrt(mean_squared_error_test))
print(np.sqrt(mean_squared_error_train))

fig, axs = plt.subplots(1,2,sharey=True,sharex=True)
axs = axs.ravel()
true_plot = sns.scatterplot(X_test["Overall"],y_test,ax=axs[0])
true_plot.set_title("True Plot")
pred_plot = sns.scatterplot(X_test["Overall"],mse_model_y_test_predictions)
pred_plot.set_title("Predicted Plot")
plt.show()
