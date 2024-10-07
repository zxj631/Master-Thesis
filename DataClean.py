import pandas as pd


def process_population_data(male_file, female_file, output_file):
    male_data = pd.read_excel(male_file, sheet_name=0, skiprows=16)
    female_data = pd.read_excel(female_file, sheet_name=0, skiprows=16)

    selected_columns = [col for col in male_data.columns if isinstance(col, int) or col in ['100+']]

    base_columns = ['Region, subregion, country or area *', 'Location code', 'Year']
    male_data = male_data[base_columns + selected_columns]
    female_data = female_data[base_columns + selected_columns]

    male_data.columns = ['Country', 'Country Code', 'Year'] + [f'Age_{col}_Male' for col in selected_columns]
    female_data.columns = ['Country', 'Country Code', 'Year'] + [f'Age_{col}_Female' for col in selected_columns]

    merged_data = pd.merge(male_data, female_data, on=['Country', 'Country Code', 'Year'])

    merged_data.to_csv(output_file, index=False)

    output_file = 'pop_data/merged_population_data_by_age_and_gender_sample.csv'
    process_population_data('pop_data/WPP2022_POP_F01_2_POPULATION_SINGLE_AGE_MALE.xlsx',
                            'pop_data/WPP2022_POP_F01_3_POPULATION_SINGLE_AGE_FEMALE.xlsx', output_file)

    print(f"Data has been saved to {output_file}")


def process_education_data():
    # Load the dataset
    file_path = 'pop_data/Educational attainment by level of education, cumulative (% population 25+).csv'
    df = pd.read_csv(file_path)

    # Split the 'Disaggregation' column to extract 'Gender' and 'Education Level'
    df[['Education Level', 'Gender']] = df['Disaggregation'].str.extract(r'(.*), (.*)')

    # Define a mapping to standardize education levels
    level_mapping = {
        'At least completed primary': 'completed primary',
        'At least completed lower secondary': 'completed lower secondary',
        'At least completed upper secondary': 'completed upper secondary',
        'At least completed post-secondary': 'completed post-secondary',
        'At least completed short-cycle tertiary': 'completed short-cycle tertiary',
        "At least Bachelor's or equivalent": "Bachelor's or equivalent",
        "At least Master's or equivalent": "Master's or equivalent",
        "At least Doctoral or equivalent": "Doctoral or equivalent"
    }

    # Apply mapping, retaining original values for unmapped items
    df['Education Level'] = df['Education Level'].map(level_mapping).fillna(df['Education Level'])

    # Pivot the DataFrame to have one row per Country, Year, and Gender
    df_pivoted = df.pivot_table(index=['Country Name', 'Country Code', 'Year', 'Gender'],
                                columns='Education Level', values='Value', aggfunc='first').reset_index()

    # Check the generated columns from pivot table
    print("Generated columns:", df_pivoted.columns.tolist())

    # Rename columns dynamically based on the pivot table result
    df_pivoted.columns.name = None

    # Expected base columns
    base_columns = ['Country Name', 'Country Code', 'Year', 'Gender']
    # Dynamic columns from pivot table
    pivot_columns = df_pivoted.columns[len(base_columns):]

    # Combine base columns with dynamic pivot columns
    df_pivoted.columns = base_columns + list(pivot_columns)

    # Export the formatted data to a CSV file
    output_path = 'pop_data/Formatted_Educational_Attainment_Data.csv'
    df_pivoted.to_csv(output_path, index=False)

    print(f"Data has been saved to {output_path}")

def merge_population_data():
    # Load the two datasets
    education_data_path = 'pop_data/Formatted_Educational_Attainment_Data.csv'
    population_data_path = 'pop_data/merged_population_data_by_age_and_gender_sample.csv'

    education_df = pd.read_csv(education_data_path)
    population_df = pd.read_csv(population_data_path, low_memory=False)

    # Ensure 'Country Name' and 'Country Code' columns are of the same type
    education_df['Country Name'] = education_df['Country Name'].astype(str)

    # Check if 'Country Code' exists in education_df
    if 'Country Code' in education_df.columns:
        education_df['Country Code'] = education_df['Country Code'].astype(str)
    else:
        education_df['Country Code'] = None  # If 'Country Code' is missing, create an empty column

    population_df['Country'] = population_df['Country'].astype(str)

    # Convert 'Year' to numeric and handle non-numeric values by coercing them to NaN
    education_df['Year'] = pd.to_numeric(education_df['Year'], errors='coerce')
    population_df['Year'] = pd.to_numeric(population_df['Year'], errors='coerce')

    # Drop rows with NaN values in the 'Year' column to ensure consistent merging
    education_df.dropna(subset=['Year'], inplace=True)
    population_df.dropna(subset=['Year'], inplace=True)

    # Convert 'Year' to integer after handling NaNs
    education_df['Year'] = education_df['Year'].astype(int)
    population_df['Year'] = population_df['Year'].astype(int)

    # Merge the two datasets on 'Country Name' and 'Year' using an outer join to keep all records
    merged_df = pd.merge(
        education_df, population_df,
        left_on=['Country Name', 'Year'], right_on=['Country', 'Year'],
        how='outer', indicator=True
    )

    # Fill missing 'Country Name' from the appropriate side
    merged_df['Country Name'] = merged_df['Country Name'].combine_first(merged_df['Country'])

    # If both 'Country Code_x' and 'Country Code_y' exist, drop 'Country Code_y' and rename 'Country Code_x' to 'Country Code'
    if 'Country Code_x' in merged_df.columns:
        merged_df['Country Code'] = merged_df['Country Code_x']  # Use 'Country Code_x' as 'Country Code'
        merged_df.drop(columns=['Country Code_x', 'Country Code_y'], inplace=True,
                       errors='ignore')  # Drop 'Country Code_y' if exists
    else:
        # If 'Country Code' already exists, do nothing
        pass

    # Drop duplicate 'Country' column from population_df after merge
    merged_df.drop(columns=['Country'], inplace=True)

    # Reorder columns to place 'Country Code' immediately after 'Country Name'
    cols = merged_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Country Code')))
    merged_df = merged_df[cols]

    # Export the merged data to a CSV file
    output_path = 'pop_data/Merged_Educational_Attainment_and_Population_Data.csv'
    merged_df.to_csv(output_path, index=False)

    print(f"The merged data has been successfully exported to the CSV file: {output_path}")

def correct_spelling():
    # Load the dataset
    merged_data_path = 'pop_data/Merged_Educational_Attainment_and_Population_Data.csv'
    merged_df = pd.read_csv(merged_data_path)

    country_code_path = 'pop_data/Country_Code.csv'
    country_code_df = pd.read_csv(country_code_path, encoding='ISO-8859-1')

    # Standardize country names to lowercase and strip any extra spaces in both datasets
    merged_df['Country Name'] = merged_df['Country Name'].str.lower().str.strip()
    country_code_df['Country'] = country_code_df['Country'].str.lower().str.strip()

    # Dictionary to map incorrect or duplicate names to a standard name based on country code data
    correction_dict = {
        'australia/new zealand': 'australia and new zealand',
        'latin america and the caribbean': 'latin america & caribbean',
        'congo, rep.': 'congo',
        'st. vincent and the grenadines': 'saint vincent and the grenadines',
        'st. kitts and nevis': 'saint kitts and nevis',
        'curacao': 'curaçao',
        'lao pdr': 'laos',
        'congo, dem. rep.': 'democratic republic of the congo',
        'gambia, the': 'gambia',
        'bahamas, the': 'bahamas',
        'kyrgyz republic': 'kyrgyzstan',
        'china, macao sar': 'macao',
        'macao sar, china': 'macao',
        'china, hong kong sar': 'hong kong',
        'hong kong sar, china': 'hong kong',
        'egypt, arab rep.': 'egypt',
        'venezuela (bolivarian republic of)': 'venezuela',
        'venezuela, rb': 'venezuela',
        'united states of america': 'united states',
        'slovak republic': 'slovakia',
        'st. lucia': 'saint lucia',
        'turkiye': 'turkey',
        'türkiye': 'turkey',
        'british virgin islands': 'virgin islands (british)',
        'virgin islands (u.s.)': 'virgin islands (u.s.)',
        "côte d'ivoire": "cote d'ivoire",
        'curaçao': 'curacao',
        'Kosovo (under UNSC res. 1244)': 'Kosovo',
        'china, taiwan province of china': 'taiwan',
        # Add more corrections as needed based on domain knowledge or the country code list
    }

    # Apply the corrections
    merged_df['Country Name'] = merged_df['Country Name'].replace(correction_dict)

    # Remove the original 'Country Code' column if it exists
    if 'Country Code' in merged_df.columns:
        merged_df.drop(columns=['Country Code'], inplace=True)

    # Merge the corrected merged_df with the country_code_df to add the Alpha-2, Alpha-3, and Numeric codes
    merged_corrected_df = pd.merge(
        merged_df,
        country_code_df,
        how='left',
        left_on='Country Name',
        right_on='Country'
    )

    # Drop the redundant 'Country' column from country_code_df
    merged_corrected_df.drop(columns=['Country', '_merge'], inplace=True)

    # Reorder columns to place the new codes immediately after 'Country Name'
    cols = merged_corrected_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Alpha-2 code')))
    cols.insert(2, cols.pop(cols.index('Alpha-3 code')))
    cols.insert(3, cols.pop(cols.index('Numeric')))
    merged_corrected_df = merged_corrected_df[cols]

    # Export the corrected data to a new CSV file
    output_path = 'pop_data/Corrected_Educational_Attainment_and_Population_Data.csv'
    merged_corrected_df.to_csv(output_path, index=False)

    print(f"The corrected data has been successfully exported to the CSV file: {output_path}")


def add_population_columns(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 获取所有与男性和女性相关的列
    male_columns = [col for col in df.columns if '_Male' in col]
    female_columns = [col for col in df.columns if '_Female' in col]

    # 计算男性总人数
    df['Male_Total'] = df[male_columns].sum(axis=1)

    # 计算女性总人数
    df['Female_Total'] = df[female_columns].sum(axis=1)

    # 计算总人数（男性 + 女性）
    df['Total_Population'] = df['Male_Total'] + df['Female_Total']

    # 保存带有新列的数据
    df.to_csv(output_file, index=False)

    print(f"The file with added population columns has been successfully saved to: {output_file}")


def merge_duplicate_countries(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确定人口数据和教育数据的列
    population_columns = [col for col in df.columns if col.startswith('Age_')]  # 人口相关的列
    key_columns = ['Country Name', 'Year']  # 国家和年份是关键列

    # 将人口数据中的 NaN 替换为 0
    df[population_columns] = df[population_columns].fillna(0)

    # 强制将人口数据列转换为数值类型
    df[population_columns] = df[population_columns].apply(pd.to_numeric, errors='coerce')

    # 计算每个国家和年份的人口总和
    population_sums = df.groupby(key_columns)[population_columns].sum().reset_index()

    # 将汇总结果覆盖原始的人口数据列，而不加后缀
    for col in population_columns:
        df[col] = df.groupby(key_columns)[col].transform('sum')

    # 不再去除重复行，只覆盖人口列的数据
    # 保存合并后的数据
    df.to_csv(output_file, index=False)

    print(f"Population data has been merged with education data and saved to {output_file}")

def fill_missing_education_data(population_df, region_mapping_df):
    """
    填充人口数据集中缺失的教育水平数据。

    :param population_df: 人口数据集 (pandas DataFrame)
    :param region_mapping_df: 国家与地区映射表 (pandas DataFrame)
    :return: 填充后的人口数据集 (pandas DataFrame)
    """
    # 合并地区信息
    population_df = population_df.merge(region_mapping_df, on='Country Name', how='left')

    # 获取教育水平列
    education_columns = ['Bachelor\'s or equivalent', 'Doctoral or equivalent', 'Master\'s or equivalent',
                         'completed lower secondary', 'completed post-secondary', 'completed primary',
                         'completed short-cycle tertiary', 'completed upper secondary']

    # 初始化一个新的 DataFrame 来存储填充后的数据
    filled_population_df = pd.DataFrame()

    # 获取所有国家列表
    countries = population_df['Country Name'].unique()

    for country in countries:
        country_df = population_df[population_df['Country Name'] == country]
        genders = country_df['Gender'].unique()

        for gender in genders:
            gender_df = country_df[country_df['Gender'] == gender]
            # 按年份排序
            gender_df = gender_df.sort_values('Year')

            # 对于每个教育水平，填充缺失值
            for edu_col in education_columns:
                # 如果该列全为空，则尝试使用地区平均值填充
                if gender_df[edu_col].isnull().all():
                    # 获取该地区、性别、教育水平的数据
                    region = gender_df['Region'].iloc[0]
                    if pd.isnull(region):
                        continue  # 无法确定地区，跳过

                    region_data = population_df[(population_df['Region'] == region) &
                                                (population_df['Gender'] == gender)]
                    # 计算地区平均值
                    region_mean = region_data.groupby('Year')[edu_col].mean()

                    # 将地区平均值合并到当前数据
                    gender_df[edu_col] = gender_df['Year'].map(region_mean)

                else:
                    # 使用插值填充缺失值
                    gender_df[edu_col] = gender_df[edu_col].interpolate(method='linear', limit_direction='both')

                    # 如果仍有缺失值，使用最近值填充
                    gender_df[edu_col].fillna(method='ffill', inplace=True)
                    gender_df[edu_col].fillna(method='bfill', inplace=True)

                    # 如果仍有缺失值，使用全局平均值
                    if gender_df[edu_col].isnull().any():
                        global_mean = population_df.groupby('Year')[edu_col].mean()
                        gender_df[edu_col].fillna(gender_df['Year'].map(global_mean), inplace=True)

            # 将填充后的数据添加到结果 DataFrame 中
            filled_population_df = pd.concat([filled_population_df, gender_df], ignore_index=True)

    return filled_population_df


if __name__ == '__main__':
    # merge_population_data()
    # correct_spelling()
    # merge_duplicate_countries('pop_data/Corrected_Educational_Attainment_and_Population_Data.csv','pop_data/Corrected_Educational_Attainment_and_Population_Data.csv')
    add_population_columns('pop_data/Corrected_Educational_Attainment_and_Population_Data.csv','pop_data/Corrected_Educational_Attainment_and_Population_Data.csv')
