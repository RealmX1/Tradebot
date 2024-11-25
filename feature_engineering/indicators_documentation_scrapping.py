from bs4 import BeautifulSoup
import pandas as pd
import requests
import json

def get_indicators_url_from_html():
    base_url = 'https://python.stockindicators.dev'
    # import html content from indicators_documentation_html.txt
    with open('feature_engineering/indicators_documentation_html.txt', 'r') as file:
        html_content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Initialize a list to store extracted data
    data = []

    # Find all h2 headers and their corresponding ul lists
    for h2 in soup.find_all('h2'):
        indicator_type = h2.text.strip()  # Get the text of the h2 tag
        ul = h2.find_next('ul')  # Find the next ul tag after the h2
        if ul:
            for li in ul.find_all('li'):  # Iterate over all li tags in the ul
                a_tag = li.find('a')  # Find the anchor tag in the li
                if a_tag:
                    name = a_tag.text.strip()  # Get the indicator name
                    href = a_tag['href'].strip()  # Get the URL
                    data.append({'type': indicator_type, 'name': name, 'url': base_url + href})

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    print(df)

    # Save the DataFrame to a CSV file if needed
    df.to_csv('feature_engineering/stock_indicators.csv', index=False)
    
    return df

def get_documentation_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request fails
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract function definition
    definition = None
    h1 = soup.find('h1')  # Find the first H1 (assumes the title is always the function name section)
    if h1:
        blockquote = h1.find_next('blockquote')  # Find the blockquote after H1
        if blockquote:
            strong_div = blockquote.find('strong')
            if strong_div:
                function_name = strong_div.text.strip()
            em_div = blockquote.find('em')  # Find <em> inside the blockquote
            if em_div:
                inputs = em_div.text.strip()
    
    definition = f"{function_name}({inputs})"
    
    # Extract parameters
    parameters = []
    h2_parameters = soup.find('h2', id='parameters')  # Find the H2 with id 'parameters'
    if h2_parameters:
        table = h2_parameters.find_next('table')  # Find the next table after the H2
        if table:
            rows = table.find('tbody').find_all('tr')  # Get all rows in the table
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 3:  # Ensure the table row has three columns
                    param_name = cols[0].text.strip()
                    param_type = cols[1].text.strip()
                    param_notes = cols[2].text.strip()
                    parameters.append({
                        'name': param_name,
                        'type': param_type,
                        'notes': param_notes
                    })
    
    # Extract return
    return_info = []
    h2_return = soup.find('h2', id='returns')  # Find the H2 with id 'return'
    if h2_return:
        h3_result = h2_return.find_next('h3', id=lambda x: x and x.endswith('result'))  # Find the next H3 with id ending in 'result'
        if h3_result:
            result_table = h3_result.find_next('table')  # Find the table after this H3
            if result_table:
                rows = result_table.find('tbody').find_all('tr')  # Get all rows in the table
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) == 3:  # Ensure the table row has three columns
                        return_name = cols[0].text.strip()
                        return_type = cols[1].text.strip()
                        return_notes = cols[2].text.strip()
                        return_info.append({
                            'name': return_name,
                            'type': return_type,
                            'notes': return_notes
                        })
    
    # Combine extracted details into a dictionary
    result = {
        'function_definition': definition,
        'parameters': parameters,
        'return': return_info
    }
    
    return result

def get_documentation_from_df(df):
    for index, row in df.iterrows():
        name = row['name']
        url = row['url']
        result = get_documentation_from_url(url)
        # add the elements in result to the df; add the new columns if not exist
        for key, value in result.items():
            if key not in df.columns:
                df[key] = None
            df.loc[index, key] = json.dumps(value)
            
    df.to_csv('feature_engineering/stock_indicators_documentation.csv', index=False)
        
    return df

def main():
    # df = get_indicators_url_from_html()
    # documentation_df = get_documentation_from_df(df)
    
    documentation_df = pd.read_csv('feature_engineering/stock_indicators_documentation.csv')
    # select two of each type and create a new df
    selected_df = documentation_df.groupby('type').head(2)
    selected_df.to_csv('feature_engineering/stock_indicators_documentation_selected.csv', index=False)
    

if __name__ == '__main__':
    main()