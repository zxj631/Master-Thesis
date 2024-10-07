import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import necessary functions from Weighting2.py
from Weighting2 import (
    calculate_weighted_age_gender_parallel,
    calculate_weighted_education_parallel,
    calculate_weighted_gender_parallel,
    calculate_overall_weight,
    clean_country_name,
)

# Set page title
st.title("Dataset Proportion Adjustment Tool")

# Upload user dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Try reading the file with different encodings
    try:
        user_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        user_data = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception as e:
        st.error(f"Unable to read the file: {e}")
        st.stop()

    # Display the first few rows of the user dataset
    st.write("Preview of your dataset:")
    st.dataframe(user_data.head())

    # Get all column names
    all_columns = user_data.columns.tolist()

    # Default select columns containing keywords, but allow users to select any column
    country_column = st.selectbox("Select country column", all_columns,
                                  index=next((i for i, col in enumerate(all_columns) if 'country' in col.lower()), 0),
                                  key='country_column')
    year_column = st.selectbox("Select year column", all_columns,
                               index=next((i for i, col in enumerate(all_columns) if 'year' in col.lower() or 'date' in col.lower()), 0),
                               key='year_column')
    # Make gender column optional
    gender_column = st.selectbox("Select gender column", all_columns,
                                 index=next((i for i, col in enumerate(all_columns) if 'gender' in col.lower()), -1) + 1,
                                 key='gender_column')
    # Make age column optional
    age_column = st.selectbox("Select age column", ['Do not use age column'] + all_columns,
                              index=next((i for i, col in enumerate(all_columns) if 'age' in col.lower()), -1) + 1,
                              key='age_column')
    # Make education column optional
    education_column = st.selectbox("Select education column", ['Do not use education column'] + all_columns,
                                    index=next((i for i, col in enumerate(all_columns) if 'edu' in col.lower()), -1) + 1,
                                    key='education_column')

    # Single-choice selection for weight calculation
    weight_options = ['Age and Gender Weight', 'Gender Weight', 'Education Level Weight', 'Overall Weight']
    selected_weight = st.radio("Select the weight type to calculate", weight_options, key='selected_weight')

    # Education level mapping (if needed)
    education_mapping_df = None
    if selected_weight in ['Education Level Weight', 'Overall Weight'] and education_column != 'Do not use education column':
        # Get unique values from user's education column
        user_education_levels = user_data[education_column].unique()
        st.write("Detected the following education levels:")
        st.write(user_education_levels)

        # Define population education level options
        population_education_levels = [
            "completed primary",
            "completed lower secondary",
            "completed upper secondary",
            "completed post-secondary",
            "Bachelor's or equivalent",
            "Master's or equivalent",
            "Doctoral or equivalent",
            "Unknown"
        ]

        # Create mapping dictionary
        education_mapping = {}
        st.write("Please select the corresponding population education level for each of your education levels:")
        for user_level in user_education_levels:
            option = st.selectbox(
                f"Population education level corresponding to '{user_level}'",
                population_education_levels,
                key=f"edu_map_{user_level}"
            )
            education_mapping[user_level] = option

        # Build education mapping DataFrame
        education_mapping_df = pd.DataFrame({
            'user education name': list(education_mapping.keys()),
            'education name': list(education_mapping.values())
        })

        # Assign an education level number to each population education level
        education_level_mapping = {
            "completed primary": 1,
            "completed lower secondary": 2,
            "completed upper secondary": 3,
            "completed post-secondary": 4,
            "Bachelor's or equivalent": 5,
            "Master's or equivalent": 6,
            "Doctoral or equivalent": 7,
            "Unknown": -1
        }
        education_mapping_df['education level'] = education_mapping_df['education name'].map(education_level_mapping)

    # Add a button to start calculation
    if st.button('Start Calculation', key='start_calculation'):
        # Read population dataset
        try:
            population_data = pd.read_csv('pop_data/Corrected_Educational_Attainment_and_Population_Data.csv',
                                          encoding='ISO-8859-1')
        except UnicodeDecodeError:
            population_data = pd.read_csv('pop_data/Corrected_Educational_Attainment_and_Population_Data.csv',
                                          encoding='utf-8')
        except Exception as e:
            st.error(f"Unable to read population dataset: {e}")
            st.stop()

        # Preprocess population dataset
        population_data['Alpha-3 code'] = population_data['Alpha-3 code'].str.strip()
        if 'Cleaned Country Name' not in population_data.columns:
            population_data['Cleaned Country Name'] = population_data['Country Name'].apply(clean_country_name)

        # Create a copy of the user dataset for results
        result_data = user_data.copy()

        # Start calculation based on selected weight
        if selected_weight == 'Age and Gender Weight':
            # Need both age and gender columns
            missing_columns = []
            if age_column == 'Do not use age column':
                missing_columns.append("age column")
            if gender_column == 'Do not use gender column':
                missing_columns.append("gender column")
            if missing_columns:
                st.error(f"Please select {', '.join(missing_columns)} to calculate Age and Gender Weight.")
            else:
                with st.spinner('Calculating age and gender weight...'):
                    try:
                        result_data = calculate_weighted_age_gender_parallel(
                            result_data, population_data, country_column, year_column, gender_column, age_column
                        )
                        st.success('Age and gender weight calculation completed.')
                    except Exception as e:
                        st.error(f"Error calculating age and gender weight: {e}")

        elif selected_weight == 'Gender Weight':
            if gender_column == 'Do not use gender column':
                st.error("Please select a gender column to calculate Gender Weight.")
            else:
                with st.spinner('Calculating gender weight...'):
                    try:
                        result_data = calculate_weighted_gender_parallel(
                            result_data, population_data, country_column, year_column, gender_column
                        )
                        st.success('Gender weight calculation completed.')
                    except Exception as e:
                        st.error(f"Error calculating gender weight: {e}")

        elif selected_weight == 'Education Level Weight':
            # Need education column
            if education_column == 'Do not use education column':
                st.error("Please select an education column to calculate Education Level Weight.")
            elif education_mapping_df is not None:
                with st.spinner('Calculating education level weight...'):
                    try:
                        result_data = calculate_weighted_education_parallel(
                            result_data, population_data, education_mapping_df, country_column, year_column,
                            gender_column if gender_column != 'Do not use gender column' else None,
                            education_column
                        )
                        st.success('Education level weight calculation completed.')
                    except Exception as e:
                        st.error(f"Error calculating education level weight: {e}")
            else:
                st.warning("Please complete the education level mapping to calculate Education Level Weight.")

        elif selected_weight == 'Overall Weight':
            # Need age, gender, and education columns
            missing_columns = []
            if age_column == 'Do not use age column':
                missing_columns.append("age column")
            if gender_column == 'Do not use gender column':
                missing_columns.append("gender column")
            if education_column == 'Do not use education column':
                missing_columns.append("education column")
            if missing_columns:
                st.error(f"Please select {', '.join(missing_columns)} to calculate Overall Weight.")
            elif education_mapping_df is None:
                st.warning("Please complete the education level mapping to calculate Overall Weight.")
            else:
                with st.spinner('Calculating overall weight...'):
                    try:
                        # First calculate age and gender weight
                        result_data = calculate_weighted_age_gender_parallel(
                            result_data, population_data, country_column, year_column, gender_column, age_column
                        )
                        # Then calculate education level weight
                        result_data = calculate_weighted_education_parallel(
                            result_data, population_data, education_mapping_df, country_column, year_column,
                            gender_column, education_column
                        )
                        # Now calculate overall weight
                        result_data = calculate_overall_weight(
                            result_data, ['Age Gender Weight', 'Education Weight']
                        )
                        st.success('Overall weight calculation completed.')
                    except Exception as e:
                        st.error(f"Error calculating overall weight: {e}")

        else:
            st.error("Invalid weight type selected.")

        # Store result_data in session state
        st.session_state['result_data'] = result_data

        # Display the dataset with new weight columns
        st.write("Preview of the dataset after calculation:")
        st.dataframe(result_data.head())

        # Provide download option
        csv = result_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download the new dataset with weights",
            data=csv,
            file_name="weighted_user_data.csv",
            mime="text/csv",
            key='download_button'
        )

    # -------------------- Analysis and Filtering Section --------------------
    # Only show analysis section if result_data is available
    if 'result_data' in st.session_state:
        st.header("Data Analysis")

        result_data = st.session_state['result_data']  # Retrieve result_data from session state

        # --- Filtering Options ---
        st.subheader("Data Filtering")

        # Country filter
        countries = result_data[country_column].dropna().unique()
        countries = sorted(countries)  # Sort country names alphabetically
        selected_countries = st.multiselect("Filter by country", options=countries, default=countries,
                                            key='analysis_filter_country')

        # Apply filters
        filtered_data = result_data.copy()
        filtered_data = filtered_data[filtered_data[country_column].isin(selected_countries)]

        # Apply filters to user_data for before weighting analysis
        filtered_user_data = user_data.copy()
        filtered_user_data = filtered_user_data[filtered_user_data[country_column].isin(selected_countries)]

        # Gender filter (if gender column is selected)
        if gender_column != 'Do not use gender column':
            genders = result_data[gender_column].dropna().unique()
            genders = sorted(genders)  # Sort genders
            selected_genders = st.multiselect("Filter by gender", options=genders, default=genders,
                                              key='analysis_filter_gender')

            # Apply gender filter
            filtered_data = filtered_data[filtered_data[gender_column].isin(selected_genders)]
            filtered_user_data = filtered_user_data[filtered_user_data[gender_column].isin(selected_genders)]

        # Age filter (if age column is selected)
        if age_column != 'Do not use age column':
            min_age = int(result_data[age_column].min())
            max_age = int(result_data[age_column].max())
            selected_age_range = st.slider("Filter by age range", min_value=min_age, max_value=max_age,
                                           value=(min_age, max_age), key='analysis_filter_age')

            # Apply age filter
            filtered_data = filtered_data[
                (filtered_data[age_column] >= selected_age_range[0]) &
                (filtered_data[age_column] <= selected_age_range[1])
                ]
            filtered_user_data = filtered_user_data[
                (filtered_user_data[age_column] >= selected_age_range[0]) &
                (filtered_user_data[age_column] <= selected_age_range[1])
                ]

        # Education filter (if education column is selected)
        if education_column != 'Do not use education column':
            educations = result_data[education_column].dropna().unique()
            educations = sorted(educations)  # Sort education levels
            selected_educations = st.multiselect("Filter by education level", options=educations, default=educations,
                                                 key='analysis_filter_education')

            # Apply education filter
            filtered_data = filtered_data[filtered_data[education_column].isin(selected_educations)]
            filtered_user_data = filtered_user_data[filtered_user_data[education_column].isin(selected_educations)]

        st.write(f"Filtered data has {len(filtered_data)} records.")
        st.write(f"Filtered data before weighting has {len(filtered_user_data)} records.")

        # For numeric data, we can create bins or plot histograms
        # Here, we will create bins for better visualization in bar charts
        num_bins = st.slider("Select number of bins for numeric data", min_value=0, max_value=100,
                             value=10, key='num_bins')

        # --- Analysis Options ---

        # Allow the user to select a variable to analyze
        analysis_columns = filtered_data.columns.tolist()
        selected_analysis_column = st.selectbox("Select a variable to analyze", analysis_columns,
                                                key='analysis_variable')

        # Allow the user to select the type of chart
        chart_types = ['Pie Chart', 'Bar Chart']
        selected_chart_type = st.selectbox("Select the type of chart", chart_types, key='analysis_chart_type')

        # Add a button to start analysis
        if st.button('Start Analysis', key='start_analysis'):
            # Prepare data for plotting
            # Check if the weight column exists
            if selected_weight == 'Overall Weight':
                weight_column = 'Overall Weight'
            elif selected_weight == 'Age and Gender Weight':
                weight_column = 'Age Gender Weight'
            elif selected_weight == 'Gender Weight':
                weight_column = 'Gender Weight'
            elif selected_weight == 'Education Level Weight':
                weight_column = 'Education Weight'
            else:
                weight_column = None

            if weight_column not in filtered_data.columns:
                st.error("Weight column not found in the data.")
            else:
                if filtered_user_data.empty:
                    st.error("No data available before weighting after applying filters.")
                else:
                    # --- Before Weighting Statistics ---
                    st.subheader("Statistics Before Weighting")

                    # Check if the analysis column is numeric
                    if pd.api.types.is_numeric_dtype(filtered_user_data[selected_analysis_column]):
                        valid_user_data = filtered_user_data[selected_analysis_column].dropna()
                        if valid_user_data.empty:
                            st.write(
                                "No valid data available for calculation before weighting after removing missing values.")
                        else:
                            mean_before = valid_user_data.mean()
                            median_before = valid_user_data.median()
                            st.write(f"Mean of {selected_analysis_column}: {mean_before}")
                            st.write(f"Median of {selected_analysis_column}: {median_before}")
                    else:
                        st.write(
                            f"The selected variable '{selected_analysis_column}' is not numeric, so mean and median cannot be calculated.")

                    # --- After Weighting Statistics ---
                    st.subheader("Statistics After Weighting")
                    if pd.api.types.is_numeric_dtype(filtered_data[selected_analysis_column]):
                        valid_data = filtered_data[[selected_analysis_column, weight_column]].dropna()
                        if valid_data.empty:
                            st.write(
                                "No valid data available for calculation after weighting after removing missing values.")
                        else:
                            data_values = valid_data[selected_analysis_column].values
                            weight_values = valid_data[weight_column].values

                            weighted_mean = np.average(data_values, weights=weight_values)


                            def weighted_median(data, weights):
                                sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
                                cumulative_weight = np.cumsum(sorted_weights)
                                cutoff = cumulative_weight[-1] / 2.0
                                return sorted_data[np.searchsorted(cumulative_weight, cutoff)]


                            weighted_median_value = weighted_median(data_values, weight_values)
                            st.write(f"Weighted Mean of {selected_analysis_column}: {weighted_mean}")
                            st.write(f"Weighted Median of {selected_analysis_column}: {weighted_median_value}")

                            # Calculate differences
                            mean_difference = weighted_mean - mean_before
                            median_difference = weighted_median_value - median_before
                            st.write(f"Difference in Mean: {mean_difference}")
                            st.write(f"Difference in Median: {median_difference}")

                            # Calculate percentage changes
                            mean_percentage_change = (mean_difference / mean_before) * 100 if mean_before != 0 else 0
                            median_percentage_change = (
                                                                   median_difference / median_before) * 100 if median_before != 0 else 0
                            st.write(f"Percentage Change in Mean: {mean_percentage_change:.2f}%")
                            st.write(f"Percentage Change in Median: {median_percentage_change:.2f}%")
                    else:
                        st.write(
                            f"The selected variable '{selected_analysis_column}' is not numeric, so weighted mean and median cannot be calculated.")

                    # --- Plotting ---
                    # Determine if the analysis column is numeric or categorical
                    if pd.api.types.is_numeric_dtype(filtered_data[selected_analysis_column]):
                        # Numeric data processing
                        st.subheader("Numeric Data Plotting")

                        # Prepare data for plotting


                        # Create bins for before weighting data
                        if num_bins > 0:
                            filtered_user_data['Binned'] = pd.cut(filtered_user_data[selected_analysis_column],
                                                                  bins=num_bins)
                            data_counts_before = filtered_user_data['Binned'].value_counts().sort_index()

                            # Create bins for after weighting data
                            filtered_data['Binned'] = pd.cut(filtered_data[selected_analysis_column], bins=num_bins)
                            data_counts_after = filtered_data.groupby('Binned')[weight_column].sum()
                        else:
                            data_counts_before = filtered_user_data[selected_analysis_column].value_counts().sort_index()
                            data_counts_after = filtered_data.groupby(selected_analysis_column)[weight_column].sum()

                        # Plot before weighting
                        st.subheader("Before Weighting")
                        fig1, ax1 = plt.subplots()

                        if selected_chart_type == 'Pie Chart':
                            # Pie charts are not ideal for continuous numeric data but can be shown for binned data
                            pie_data = data_counts_before[data_counts_before > 0]
                            if pie_data.empty:
                                st.write(
                                    "No data available for plotting before weighting after removing missing values.")
                            else:
                                ax1.pie(pie_data.values, labels=pie_data.index.astype(str), autopct='%1.1f%%')
                                ax1.axis('equal')
                                ax1.set_title(f"Distribution of {selected_analysis_column}")
                                st.pyplot(fig1)
                        elif selected_chart_type == 'Bar Chart':
                            ax1.bar(data_counts_before.index.astype(str),
                                    data_counts_before.values,
                                    color='skyblue', edgecolor='black')
                            ax1.set_xlabel(selected_analysis_column)
                            ax1.set_ylabel('Count')
                            plt.xticks(rotation=90)
                            st.pyplot(fig1)

                        # Plot after weighting
                        st.subheader("After Weighting")
                        fig2, ax2 = plt.subplots()

                        if selected_chart_type == 'Pie Chart':
                            pie_data = data_counts_after[data_counts_after > 0]
                            if pie_data.empty:
                                st.write(
                                    "No data available for plotting after weighting after removing missing values.")
                            else:
                                ax2.pie(pie_data.values, labels=pie_data.index.astype(str), autopct='%1.1f%%')
                                ax2.axis('equal')
                                ax2.set_title(f"Weighted Distribution of {selected_analysis_column}")
                                st.pyplot(fig2)
                        elif selected_chart_type == 'Bar Chart':
                            ax2.bar(data_counts_after.index.astype(str),
                                    data_counts_after.values,
                                    color='salmon', edgecolor='black')
                            ax2.set_xlabel(selected_analysis_column)
                            ax2.set_ylabel('Weighted Count')
                            plt.xticks(rotation=90)
                            st.pyplot(fig2)
                    else:
                        # Categorical data processing
                        st.subheader("Categorical Data Plotting")

                        # Prepare data for plotting
                        # Get all categories
                        categories_before = filtered_user_data[selected_analysis_column].dropna().astype(str)
                        categories_after = filtered_data[selected_analysis_column].dropna().astype(str)
                        all_categories = sorted(set(categories_before.unique()).union(set(categories_after.unique())))

                        # Prepare counts for before weighting
                        data_counts_before = categories_before.value_counts()
                        data_counts_before = data_counts_before.reindex(all_categories, fill_value=0)

                        # Prepare counts for after weighting
                        data_counts_after = filtered_data.groupby(selected_analysis_column)[weight_column].sum()
                        data_counts_after = data_counts_after.reindex(all_categories, fill_value=0)

                        # Plot before weighting
                        st.subheader("Before Weighting")
                        fig1, ax1 = plt.subplots()

                        if selected_chart_type == 'Pie Chart':
                            # Exclude zero counts for pie charts
                            pie_data = data_counts_before[data_counts_before > 0]
                            if pie_data.empty:
                                st.write(
                                    "No data available for plotting before weighting after removing missing values.")
                            else:
                                ax1.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
                                ax1.axis('equal')
                                ax1.set_title(f"Distribution of {selected_analysis_column}")
                                st.pyplot(fig1)
                        elif selected_chart_type == 'Bar Chart':
                            ax1.bar(data_counts_before.index,
                                    data_counts_before.values,
                                    color='skyblue', edgecolor='black')
                            ax1.set_xlabel(selected_analysis_column)
                            ax1.set_ylabel('Count')
                            plt.xticks(rotation=90)
                            st.pyplot(fig1)

                        # Plot after weighting
                        st.subheader("After Weighting")
                        fig2, ax2 = plt.subplots()

                        if selected_chart_type == 'Pie Chart':
                            # Exclude zero counts for pie charts
                            pie_data = data_counts_after[data_counts_after > 0]
                            if pie_data.empty:
                                st.write(
                                    "No data available for plotting after weighting after removing missing values.")
                            else:
                                ax2.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
                                ax2.axis('equal')
                                ax2.set_title(f"Weighted Distribution of {selected_analysis_column}")
                                st.pyplot(fig2)
                        elif selected_chart_type == 'Bar Chart':
                            ax2.bar(data_counts_after.index,
                                    data_counts_after.values,
                                    color='salmon', edgecolor='black')
                            ax2.set_xlabel(selected_analysis_column)
                            ax2.set_ylabel('Weighted Count')
                            plt.xticks(rotation=90)
                            st.pyplot(fig2)



else:
    st.warning("Please upload a dataset.")
