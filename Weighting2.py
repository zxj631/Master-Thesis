import re
from itertools import islice
import pandas as pd
from joblib import Parallel, delayed


def calculate_age_gender_weight(population_df, country_code, year, gender, age, user_ratio):
    """
    Calculate the ratio (weight) of the real-world proportion to the user-input proportion for a specified country code, year, gender, and age group.

    :param population_df: Population dataset (pandas DataFrame)
    :param country_code: Country code (Alpha-3 code)
    :param year: Year
    :param gender: Gender ('Male', 'Female', or None)
    :param age: Age group
    :param user_ratio: User input data ratio
    :return: Weight for age and gender
    """
    # Standardize country code and remove spaces
    country_code = country_code.strip()
    population_df['Alpha-3 code'] = population_df['Alpha-3 code'].str.strip()
    print(f'Start calculating {country_code} weight')

    # Filter data for the specified country and year
    country_data = population_df[(population_df['Alpha-3 code'] == country_code) & (population_df['Year'] == year)]

    if country_data.empty:
        return None  # Return None if no data is found for the specified country and year

    # Use vectorized calculation for age ratio
    male_age_col = f'Age_{age}_Male'
    female_age_col = f'Age_{age}_Female'

    if gender.lower() == 'male':
        # If male age group data exists, calculate the male age group ratio
        if male_age_col in country_data.columns:
            real_male_age_ratio = country_data[male_age_col].values[0] / country_data['Male_Total'].values[0] if \
                country_data['Male_Total'].values[0] > 0 else None
            return real_male_age_ratio / user_ratio if real_male_age_ratio is not None else None
    elif gender.lower() == 'female':
        # If female age group data exists, calculate the female age group ratio
        if female_age_col in country_data.columns:
            real_female_age_ratio = country_data[female_age_col].values[0] / country_data['Female_Total'].values[0] if \
                country_data['Female_Total'].values[0] > 0 else None
            return real_female_age_ratio / user_ratio if real_female_age_ratio is not None else None
    else:
        # If gender is None, calculate the total ratio for both males and females
        if male_age_col in country_data.columns and female_age_col in country_data.columns:
            total_age_population = country_data[male_age_col].values[0] + country_data[female_age_col].values[0]
            real_total_age_ratio = total_age_population / country_data['Total_Population'].values[0] if \
                country_data['Total_Population'].values[0] > 0 else None
            return real_total_age_ratio / user_ratio if real_total_age_ratio is not None else None

    return None


def calculate_age_gender_ratio(user_df, country_column, country_name, gender_column, age_column, age, gender,
                               male_value='Male', female_value='Female'):
    """
    Calculate the ratio of the number of people in a specified country, gender, and age group to the total number of people in the user input dataset.

    :param user_df: User input dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param country_name: Name of the country to calculate
    :param gender_column: Column name representing gender in the user dataset
    :param age_column: Column name representing age in the user dataset
    :param age: Age group to calculate
    :param gender: Gender ('Male', 'Female', or other)
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :return: Ratio of the specified group
    """
    print(f'Start calculating {country_name}')
    # First filter by country and age to reduce the dataset scope
    filtered_df = user_df[(user_df[country_column] == country_name) & (user_df[age_column] == age)]

    # Further filter by gender
    if gender == male_value:
        user_filtered = filtered_df[filtered_df[gender_column] == male_value]
    elif gender == female_value:
        user_filtered = filtered_df[filtered_df[gender_column] == female_value]
    else:
        user_filtered = filtered_df

    # Calculate total population and selected population
    total_population = len(user_df[user_df[country_column] == country_name])
    selected_population = len(user_filtered)

    return selected_population / total_population if total_population > 0 else None


def calculate_single_ratio_and_weight(group, user_df, population_df, country_column, year_column, gender_column,
                                      age_column, male_value, female_value):
    country_name = group[country_column].iloc[0]
    year = group[year_column].iloc[0]
    gender = group[gender_column].iloc[0]
    age = group[age_column].iloc[0]

    country_info = find_country_info(population_df, country_name)
    if country_info is None:
        print(f"Warning: Could not find country info for '{country_name}'. Skipping this group.")
        return pd.Series([None] * len(group), index=group.index)
    else:
        country_code = country_info[-1]

    user_ratio = calculate_age_gender_ratio(user_df, country_column, country_name, gender_column, age_column, age,
                                            gender, male_value, female_value)

    if user_ratio is None:
        return pd.Series([None] * len(group), index=group.index)

    matched_population = population_df[
        (population_df['Alpha-3 code'] == country_code) & (population_df['Year'] == year)]

    if matched_population.empty:
        return pd.Series([None] * len(group), index=group.index)

    weight = calculate_age_gender_weight(matched_population, country_code, year, gender, age, user_ratio)

    # Add debug statement
    print(f"Group: Country={country_name}, Year={year}, Gender={gender}, Age={age}, Weight={weight}")

    return pd.Series([weight] * len(group), index=group.index)


def batch(iterable, batch_size):
    """
    Generator function to iterate over iterable in batches.

    :param iterable: Iterable to batch
    :param batch_size: Size of each batch
    :return: Yields lists of batched items
    """
    iterable = iter(iterable)
    while True:
        batch_iter = list(islice(iterable, batch_size))
        if not batch_iter:
            break
        yield batch_iter


def calculate_weighted_age_gender_parallel(user_df, population_df, country_column, year_column, gender_column,
                                           age_column, male_value='Male', female_value='Female', n_jobs=-1,
                                           batch_size=100):
    """
    Parallel computation of age and gender weights for each group, processed in batches to optimize efficiency.

    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param age_column: Column name representing age in the user dataset
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :param n_jobs: Number of jobs for parallel processing, default is -1 (use all cores)
    :param batch_size: Size of each batch for processing, default is 100
    :return: User dataset with added 'Age Gender Weight' column
    """
    # Use a unique index column name
    original_index_column = 'original_index_age_gender'
    if original_index_column not in user_df.columns:
        user_df = user_df.reset_index()
        user_df.rename(columns={'index': original_index_column}, inplace=True)

    user_df = convert_to_year_only(user_df, year_column)
    user_df[[country_column, year_column, gender_column, age_column]] = user_df[
        [country_column, year_column, gender_column, age_column]].fillna('Unknown')
    # Group user data by country, year, gender, and age
    grouped = user_df.groupby([country_column, year_column, gender_column, age_column])

    # Create group_list and sort by size to balance groups
    group_list = sorted([group for _, group in grouped], key=len, reverse=True)
    print(f'{len(group_list)} groups sorted.')

    # Use batch to process group_list in batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch_group, user_df, population_df, country_column, year_column, gender_column,
                               age_column, male_value, female_value)
        for batch_group in batch(group_list, batch_size)
    )

    # Concatenate the calculated weight results into a Series
    weights = pd.concat(results)

    # Return the user dataset with the added weight column
    user_df['Age Gender Weight'] = weights
    return user_df


def process_batch(batch, user_df, population_df, country_column, year_column, gender_column, age_column, male_value,
                  female_value):
    """
    Process a batch of data groups to calculate weights for each group.

    :param batch: List of groups in the batch
    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param age_column: Column name representing age in the user dataset
    :param male_value: Value representing male
    :param female_value: Value representing female
    :return: Concatenated results of weight calculations for the batch
    """
    batch_results = []
    for group in batch:
        batch_results.append(
            calculate_single_ratio_and_weight(group, user_df, population_df, country_column, year_column, gender_column,
                                              age_column, male_value, female_value))
    return pd.concat(batch_results)


def calculate_education_level_ratio(user_df, country_column, country_name, gender_column, education_column,
                                    education_level, gender, male_value='Male', female_value='Female'):
    """
    Calculate the ratio of the number of people with a specified education level in a given country and gender to the total number of people in the user input dataset.

    :param user_df: User input dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param country_name: Name of the country to calculate
    :param gender_column: Column name representing gender in the user dataset
    :param education_column: Column name representing education level in the user dataset
    :param education_level: Education level to calculate (e.g., Bachelor's or equivalent, Master's or equivalent)
    :param gender: Gender ('Male', 'Female', or None)
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :return: Ratio of the specified education level
    """
    # Filter data by country
    country_filtered_df = user_df[user_df[country_column] == country_name]

    # Filter data by gender
    if gender == male_value:
        # Filter male data
        gender_filtered_df = country_filtered_df[country_filtered_df[gender_column] == male_value]
    elif gender == female_value:
        # Filter female data
        gender_filtered_df = country_filtered_df[country_filtered_df[gender_column] == female_value]
    else:
        # If gender is not specified, use all data for the country
        gender_filtered_df = country_filtered_df

    # Filter records by specified education level
    education_filtered_df = gender_filtered_df[gender_filtered_df[education_column] == education_level]

    # Calculate the ratio of people with the specified education level to the total population in the filtered data
    total_population = len(gender_filtered_df)  # Total number of people matching the gender condition
    selected_population = len(education_filtered_df)  # Number of records matching both gender and education level

    if total_population > 0:
        education_ratio = selected_population / total_population
    else:
        education_ratio = None  # If total population is 0, the ratio cannot be calculated

    return education_ratio


def get_closest_education_ratio(df, year, gender, target_education_level):
    """
    Get the education ratio data closest to the target year.

    :param df: Population data DataFrame
    :param year: Target year
    :param gender: Gender ('male' or 'female')
    :param target_education_level: Target education level
    :return: Closest year's education ratio, or None if no data is available
    """
    # Filter data by specified gender
    filtered_df = df[df['Gender'].str.lower() == gender.lower()]

    if filtered_df.empty:
        return None

    # Check if the target education level column exists
    if target_education_level not in filtered_df.columns:
        return None

    # Extract available data for the target education level
    data = filtered_df[['Year', target_education_level]].dropna()

    if data.empty:
        return None

    # Ensure 'Year' column is of integer type
    data['Year'] = data['Year'].astype(int)

    # Calculate the difference between the target year and available years
    data['Year_diff'] = abs(data['Year'] - year)

    # Find the row with the smallest year difference
    closest_row = data.loc[data['Year_diff'].idxmin()]
    closest_ratio = closest_row[target_education_level]

    return closest_ratio


def calculate_education_weight_with_gender(population_df, country_code, year, gender, education_level,
                                           next_education_level, user_ratio, education_mapping_df):
    """
    Calculate the weight for education level with gender consideration.

    :param population_df: Population dataset (pandas DataFrame)
    :param country_code: Country code (Alpha-3 code)
    :param year: Year
    :param gender: Gender ('Male' or 'Female')
    :param education_level: Current education level
    :param next_education_level: Next education level
    :param user_ratio: User input ratio for the current education level
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :return: Education weight
    """
    # Standardize country code and gender
    country_code = country_code.strip()
    gender = gender.lower()

    # Filter data for the specified country
    country_data_all_years = population_df[(population_df['Alpha-3 code'] == country_code)]

    if country_data_all_years.empty:
        return None  # Return None if no data is found for the country

    # Exclude 'Unknown' education levels
    education_levels = education_mapping_df[education_mapping_df['education name'] != 'Unknown'][
        'education name'].tolist()

    # Create education level mapping
    education_mapping = {level: idx + 1 for idx, level in enumerate(education_levels)}

    # Use the modified function to get the education ratio
    education_ratio = get_closest_education_ratio(
        country_data_all_years, year, gender, education_level
    )

    if education_ratio is None:
        return None

    next_education_ratio = get_closest_education_ratio(
        country_data_all_years, year, gender, next_education_level
    ) if next_education_level is not None else 0

    if next_education_ratio is None:
        next_education_ratio = 0

    # Ensure the next education level ratio is not greater than the current education level ratio
    if next_education_ratio > education_ratio:
        next_education_ratio = education_ratio

    # Calculate the predicted ratio difference
    predicted_ratio = (education_ratio - next_education_ratio) / 100

    # Calculate weight
    weight = predicted_ratio / user_ratio if predicted_ratio > 0 else None

    return weight


def clean_country_name(name):
    """
    Clean the country name by removing all non-alphabetic characters.

    :param name: Country name string
    :return: Lowercase country name containing only letters
    """
    return re.sub(r'[^a-z]', '', name.lower())


def find_country_info(population_df, country_name=None, country_code=None):
    """
    Find country information based on country name or country code.

    :param population_df: Population dataset (pandas DataFrame)
    :param country_name: Name of the country
    :param country_code: Code of the country
    :return: Tuple of (Country Name, Alpha-3 Code) or None if not found
    """
    if country_name:
        country_name_cleaned = clean_country_name(country_name)
    if country_code:
        country_code = country_code.strip().lower()

    if country_code:
        result = population_df[
            (population_df['Alpha-2 code'].str.lower() == country_code) |
            (population_df['Alpha-3 code'].str.lower() == country_code) |
            (population_df['Numeric'].astype(str) == country_code)
            ]
    elif country_name:
        if 'Cleaned Country Name' not in population_df.columns:
            population_df['Cleaned Country Name'] = population_df['Country Name'].apply(clean_country_name)
        result = population_df[population_df['Cleaned Country Name'] == country_name_cleaned]
    else:
        return None

    if result.empty:
        print(
            f"Warning: Can not find country info for country_name '{country_name}' (cleaned: '{country_name_cleaned}')")
        return None

    country_name = result['Country Name'].values[0]
    alpha_3_code = result['Alpha-3 code'].values[0]

    return country_name, alpha_3_code


def get_next_education_level(current_level, education_mapping_df):
    """
    Get the name of the next education level based on the current level number.

    :param current_level: Current education level number (int)
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :return: Name of the next education level or None
    """
    next_level = current_level + 1
    next_education_row = education_mapping_df[education_mapping_df['education level'] == next_level]

    if not next_education_row.empty:
        return next_education_row['education name'].values[0]

    return None


def map_education_level(user_education_level, education_mapping_df):
    """
    Map the user's education level to the population's education level and return the education name and level number.

    :param user_education_level: Education level name in the user dataset
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :return: Tuple of (Mapped Education Level Name, Education Level Number) or (None, None)
    """
    mapped_row = education_mapping_df[education_mapping_df['user education name'] == user_education_level]

    if not mapped_row.empty:
        return mapped_row['education name'].values[0], mapped_row['education level'].values[0]

    return None, None


def calculate_single_education_ratio_and_weight(group, user_df, population_df, education_mapping_df, country_column, year_column, gender_column, education_column, education_level_number, male_value, female_value):
    """
    Calculate the education ratio and weight for a single group.

    :param group: Grouped data (pandas DataFrame)
    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param education_column: Column name representing education level in the user dataset
    :param education_level_number: Column name representing education level number in the user dataset
    :param male_value: Value representing male
    :param female_value: Value representing female
    :return: DataFrame with original indices and calculated weights
    """
    country_name = group[country_column].iloc[0]
    year = group[year_column].iloc[0]
    gender = group[gender_column].iloc[0]
    education_level = group[education_column].iloc[0]
    education_level_num = group[education_level_number].iloc[0]

    country_info = find_country_info(population_df, country_name)
    if country_info is None:
        print(f"Warning: Could not find country info for '{country_name}'. Skipping this group.")
        return pd.DataFrame({
            'original_index': group['original_index'],
            'Education Weight': [None] * len(group)
        })
    else:
        country_code = country_info[-1]

    # Calculate user ratio
    user_ratio = calculate_education_level_ratio(
        user_df, country_column, country_name, gender_column, 'Mapped Education Level', education_level,
        gender, male_value, female_value
    )

    if user_ratio is None:
        return pd.DataFrame({
            'original_index': group['original_index'],
            'Education Weight': [None] * len(group)
        })

    # Get next education level
    next_education_level = get_next_education_level(education_level_num, education_mapping_df)

    # Calculate weight
    weight = calculate_education_weight_with_gender(
        population_df, country_code, year, gender, education_level, next_education_level, user_ratio, education_mapping_df
    )

    # Return DataFrame with original indices and calculated weights
    return pd.DataFrame({
        'original_index': group['original_index'],
        'Education Weight': [weight] * len(group)
    })


def process_education_batch(batch, user_df, population_df, education_mapping_df, country_column, year_column, gender_column, education_column, education_level_number, male_value, female_value):
    """
    Process a batch of education data groups to calculate weights for each group.

    :param batch: List of groups in the batch
    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param education_column: Column name representing education level in the user dataset
    :param education_level_number: Column name representing education level number in the user dataset
    :param male_value: Value representing male
    :param female_value: Value representing female
    :return: Concatenated results of education weight calculations for the batch
    """
    batch_results = []
    for group in batch:
        result = calculate_single_education_ratio_and_weight(
            group, user_df, population_df, education_mapping_df, country_column, year_column, gender_column, education_column, education_level_number, male_value, female_value
        )
        batch_results.append(result)
    return pd.concat(batch_results)


def calculate_weighted_education_parallel(user_df, population_df, education_mapping_df, country_column, year_column,
                                          gender_column, education_column, male_value='Male', female_value='Female',
                                          n_jobs=-1, batch_size=100):
    """
    Parallel computation of education weights for each group, processed in batches to optimize efficiency.

    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param education_mapping_df: Education mapping table (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param education_column: Column name representing education level in the user dataset
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :param n_jobs: Number of jobs for parallel processing, default is -1 (use all cores)
    :param batch_size: Size of each batch for processing, default is 100
    :return: User dataset with added 'Education Weight' column
    """
    original_index_column = 'original_index_education'
    if original_index_column not in user_df.columns:
        user_df = user_df.reset_index()
        user_df.rename(columns={'index': original_index_column}, inplace=True)

    user_df = convert_to_year_only(user_df, year_column)
    user_df = user_df.reset_index()
    original_index_column = 'original_index'
    user_df.rename(columns={'index': original_index_column}, inplace=True)

    # Fill missing values in group columns
    user_df[[country_column, year_column, gender_column, education_column]] = user_df[
        [country_column, year_column, gender_column, education_column]].fillna('Unknown')

    # Map user education levels to population education levels and get education level numbers
    user_df['Mapped Education Level'], user_df['Education Level Number'] = zip(*user_df[education_column].apply(
        lambda x: map_education_level(x, education_mapping_df)
    ))

    # Filter out entries where Mapped Education Level is 'Unknown'
    user_df = user_df[(user_df['Mapped Education Level'].notnull()) & (user_df['Mapped Education Level'] != 'Unknown')]

    # Group user data by country, year, gender, mapped education level, and education level number
    grouped = user_df.groupby(
        [country_column, year_column, gender_column, 'Mapped Education Level', 'Education Level Number'])

    # Create group_list and sort by size to balance groups
    group_list = sorted([group for _, group in grouped], key=lambda x: len(x), reverse=True)
    print(f'{len(group_list)} education groups sorted.')

    # Use batch processing to handle group_list in batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_education_batch)(
            batch_group, user_df, population_df, education_mapping_df, country_column, year_column, gender_column,
            'Mapped Education Level', 'Education Level Number', male_value, female_value
        )
        for batch_group in batch(group_list, batch_size)
    )

    # Concatenate the results into a DataFrame
    weights_df = pd.concat(results)

    # When merging weights_df back into user_df, use the correct index column
    user_df = user_df.merge(weights_df, on=original_index_column, how='left')

    # Do not drop the original_index_column
    user_df.drop(columns=['Mapped Education Level', 'Education Level Number'], inplace=True)

    return user_df


def convert_to_year_only(user_df, year_column):
    """
    Convert the year column in the user dataset to only contain the year. If the year already exists, keep it unchanged.

    :param user_df: User input dataset (pandas DataFrame)
    :param year_column: Column name representing the year in the user dataset
    :return: Converted user dataset (pandas DataFrame)
    """
    # Use to_datetime to convert the year column to datetime type, replacing invalid data with NaT
    user_df[year_column] = pd.to_datetime(user_df[year_column], errors='coerce')

    # Extract the year part and replace the original date column
    user_df[year_column] = user_df[year_column].dt.year

    return user_df


def calculate_gender_ratio(user_df, country_column, country_name, gender_column, gender, male_value='Male',
                           female_value='Female'):
    """
    Calculate the ratio of the number of people of a specified gender in a given country to the total number of people in that country.

    :param user_df: User dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param country_name: Name of the country to calculate
    :param gender_column: Column name representing gender in the user dataset
    :param gender: Gender ('Male' or 'Female')
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :return: Gender ratio
    """
    # Filter data by country
    country_filtered_df = user_df[user_df[country_column] == country_name]

    # Filter data by gender
    if gender == male_value:
        gender_filtered_df = country_filtered_df[country_filtered_df[gender_column] == male_value]
    elif gender == female_value:
        gender_filtered_df = country_filtered_df[country_filtered_df[gender_column] == female_value]
    else:
        # If gender is not specified, return None
        return None

    total_population = len(country_filtered_df)
    selected_population = len(gender_filtered_df)

    if total_population > 0:
        gender_ratio = selected_population / total_population
    else:
        gender_ratio = None

    return gender_ratio


def calculate_population_gender_ratio(population_df, country_code, year, gender):
    """
    Calculate the real-world gender ratio for a specified country and year.

    :param population_df: Population dataset (pandas DataFrame)
    :param country_code: Country code (Alpha-3 code)
    :param year: Year
    :param gender: Gender ('Male' or 'Female')
    :return: Real-world gender ratio
    """
    # Filter data for the specified country and year
    country_data = population_df[(population_df['Alpha-3 code'] == country_code) & (population_df['Year'] == year)]

    if country_data.empty:
        return None

    if gender.lower() == 'male':
        male_total = country_data['Male_Total'].values[0]
        total_population = country_data['Total_Population'].values[0]
        if total_population > 0:
            gender_ratio = male_total / total_population
        else:
            gender_ratio = None
    elif gender.lower() == 'female':
        female_total = country_data['Female_Total'].values[0]
        total_population = country_data['Total_Population'].values[0]
        if total_population > 0:
            gender_ratio = female_total / total_population
        else:
            gender_ratio = None
    else:
        gender_ratio = None

    return gender_ratio


def calculate_gender_weight(population_df, country_code, year, gender, user_ratio):
    """
    Calculate the weight for a specified country, year, and gender.

    :param population_df: Population dataset (pandas DataFrame)
    :param country_code: Country code (Alpha-3 code)
    :param year: Year
    :param gender: Gender ('Male' or 'Female')
    :param user_ratio: User data gender ratio
    :return: Gender weight
    """
    real_ratio = calculate_population_gender_ratio(population_df, country_code, year, gender)
    if real_ratio is None or user_ratio is None or user_ratio == 0:
        return None

    weight = real_ratio / user_ratio

    return weight


def calculate_single_gender_ratio_and_weight(group, user_df, population_df, country_column, year_column, gender_column,
                                             male_value, female_value):
    """
    Calculate the gender ratio and weight for a single group.

    :param group: Grouped data (pandas DataFrame)
    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param male_value: Value representing male
    :param female_value: Value representing female
    :return: Series with weights for the group
    """
    country_name = group[country_column].iloc[0]
    year = group[year_column].iloc[0]
    gender = group[gender_column].iloc[0]

    country_info = find_country_info(population_df, country_name)
    if country_info is None:
        print(f"Warning: Could not find country info for '{country_name}'. Skipping this group.")
        return pd.Series([None] * len(group), index=group.index)
    else:
        country_code = country_info[-1]

    user_ratio = calculate_gender_ratio(user_df, country_column, country_name, gender_column, gender,
                                        male_value, female_value)

    if user_ratio is None:
        return pd.Series([1] * len(group), index=group.index)

    weight = calculate_gender_weight(population_df, country_code, year, gender, user_ratio)

    # Add debug information
    print(f"Group: Country={country_name}, Year={year}, Gender={gender}, Weight={weight}")

    return pd.Series([weight] * len(group), index=group.index)


def process_gender_batch(batch, user_df, population_df, country_column, year_column, gender_column, male_value,
                         female_value):
    """
    Process a batch of gender data groups to calculate weights for each group.

    :param batch: List of groups in the batch
    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param male_value: Value representing male
    :param female_value: Value representing female
    :return: Concatenated results of gender weight calculations for the batch
    """
    batch_results = []
    for group in batch:
        result = calculate_single_gender_ratio_and_weight(
            group, user_df, population_df, country_column, year_column, gender_column, male_value, female_value
        )
        batch_results.append(result)
    return pd.concat(batch_results)


def calculate_weighted_gender_parallel(user_df, population_df, country_column, year_column, gender_column,
                                       male_value='Male', female_value='Female', n_jobs=-1, batch_size=100):
    """
    Parallel computation of gender weights for each group, processed in batches to optimize efficiency.

    :param user_df: User input dataset (pandas DataFrame)
    :param population_df: Population dataset (pandas DataFrame)
    :param country_column: Column name representing the country in the user dataset
    :param year_column: Column name representing the year in the user dataset
    :param gender_column: Column name representing gender in the user dataset
    :param male_value: Value representing male, default is 'Male'
    :param female_value: Value representing female, default is 'Female'
    :param n_jobs: Number of jobs for parallel processing, default is -1 (use all cores)
    :param batch_size: Size of each batch for processing, default is 100
    :return: User dataset with added 'Gender Weight' column
    """
    original_index_column = 'original_index_gender'
    if original_index_column not in user_df.columns:
        user_df = user_df.reset_index()
        user_df.rename(columns={'index': original_index_column}, inplace=True)

    user_df = convert_to_year_only(user_df, year_column)
    user_df = user_df.reset_index()
    original_index_column = 'original_index'
    user_df.rename(columns={'index': original_index_column}, inplace=True)

    # Fill missing values in group columns
    user_df[[country_column, year_column, gender_column]] = user_df[
        [country_column, year_column, gender_column]].fillna('Unknown')

    # Group user data by country, year, and gender
    grouped = user_df.groupby([country_column, year_column, gender_column])

    # Create group_list and sort by size to balance groups
    group_list = sorted([group for _, group in grouped], key=lambda x: len(x), reverse=True)
    print(f'{len(group_list)} gender groups sorted.')

    # Use batch processing to handle group_list in batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_gender_batch)(
            batch_group, user_df, population_df, country_column, year_column, gender_column, male_value, female_value
        )
        for batch_group in batch(group_list, batch_size)
    )

    # Concatenate the results into a Series
    weights = pd.concat(results)

    # Add the weights to the user_df as a new column
    user_df['Gender Weight'] = weights

    # Return the user dataset with the added gender weight column
    return user_df


def calculate_overall_weight(user_df, weight_columns):
    """
    Calculate the overall weight by multiplying specified weight columns.

    Parameters:
    - user_df: pandas DataFrame containing user data and weight columns.
    - weight_columns: List of column names representing individual weights to multiply.

    Returns:
    - user_df: pandas DataFrame with a new column 'Overall Weight'.
    """
    # Check if all specified weight columns exist in the DataFrame
    missing_columns = [col for col in weight_columns if col not in user_df.columns]
    if missing_columns:
        raise ValueError(f"Missing weight columns: {missing_columns}")

    # Calculate the product of the specified weight columns
    user_df['Overall Weight'] = user_df[weight_columns].prod(axis=1)

    return user_df


# if __name__ == '__main__':
    # population_df = pd.read_csv('pop_data/Corrected_Educational_Attainment_and_Population_Data.csv')
    # user_df = pd.read_csv('user_data/COVIDiSTRESS_May_30_cleaned_final.csv', encoding='ISO-8859-1')
    #
    # # Preprocess population_df
    # population_df['Alpha-3 code'] = population_df['Alpha-3 code'].str.strip()
    # if 'Cleaned Country Name' not in population_df.columns:
    #     population_df['Cleaned Country Name'] = population_df['Country Name'].apply(clean_country_name)
    #
    # weight = calculate_age_gender_weight(population_df, 'USA', 2020, 'M', 25, 0.5)
    # weight_edu = calculate_education_weight_with_gender(population_df, 'USA', 2012, 'M', "Bachelor's or equivalent",
    #                                                     "Master's or equivalent", 0.5)
    # print(weight)
    # print(weight_edu)
    #
    # result = find_country_info(population_df, country_name='Canada')
    # print(result)  # Output: ('Canada', 'CAN')
    #
    # result = calculate_weighted_age_gender_parallel(user_df, population_df, 'Country', 'RecordedDate', 'Dem_gender',
    #                                                 'Dem_age')
    # print(result)
    # result.to_csv('user_data/result.csv', index=False)
    #
    # grouped = user_df.groupby(['Dem_edu'])
    # for group in grouped:
    #     print(group[0])
    #
    # edu_data = {
    #     "education name": ["Bachelor's or equivalent", "Doctoral or equivalent", "completed lower secondary",
    #                        "completed post-secondary", "completed primary", "completed upper secondary", 'Unknown'],
    #     "user education name": ['College degree, bachelor, master', 'PhD/Doctorate', 'Up to 9 years of school',
    #                             'Some College, short continuing education or equivalent', 'Up to 6 years of school',
    #                             'Up to 12 years of school', "Uninformative response"],
    #     "education level": [5, 6, 2, 4, 1, 3, -1]
    # }
    #
    # edu_df = pd.DataFrame(edu_data)
    # print(edu_df)
    #
    # result = calculate_weighted_education_parallel(user_df, population_df, edu_df, 'Country', 'RecordedDate',
    #                                                'Dem_gender', 'Dem_edu')
    # result.to_csv('user_data/result.csv', index=False)
