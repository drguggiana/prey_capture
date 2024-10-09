import requests
import paths
import json
import tkinter as tk
from collections import defaultdict
from typing import Dict, List, Union, Type

import functions_data_handling as fd

LOOKUPS = {
    'gtdate': 'date__gt',
    'ltdate': 'date__lt',
    'slug': 'slug__icontains',
    'notes': 'notes__icontains',
}


class SearchBon:
    """
    This is a class used to represent a Search object.

    Attributes:
        search_result (Dict[str, Union[str, int]]): The result of the search operation.
    """

    def __init__(self, search_result: Dict[str, Union[str, int]]):
        """
        The constructor for the SearchBon class.

        Parameters:
            search_result (Dict[str, Union[str, int]]): The result of the search operation.
        """
        self.search_result = search_result


def parse_results(searchbon_object: SearchBon) -> Dict[str, List[Union[str, int]]]:
    """
    This function parses the bonDjango search results into a dictionary.

    Parameters:
    searchbon_object (SearchBon): An instance of the SearchBon class.

    Returns:
    Dict[str, List[Union[str, int]]]: A dictionary where each key is a string and the value is a list of strings or integers.
    """

    parsed_results = defaultdict(list)
    # Iterate over the results in the searchbon_object
    for index in searchbon_object.search_result['results']:
        # For each result, iterate over its items
        for k, v in index.items():
            # Append the item value to the corresponding key in the parsed_results dictionary
            parsed_results[k].append(v)
    # Return the parsed results
    return parsed_results


def query_gui() -> Type[SearchBon]:
    """
    This function runs a small GUI to deliver a search query to the database.

    Returns:
    SearchBon: An instance of the SearchBon class with the search result.

    """
    # Create a new tkinter window
    top = tk.Tk()

    # Create an entry widget in the window
    search_box = tk.Entry(top)
    search_box.pack(side=tk.TOP)

    # Create an instance of the SearchBon class
    search_object = SearchBon

    def query_server():
        """
        This function gets the query from the search box, sends a GET request to the server,
        and stores the result in the search_object.
        """
        # Get the query from the search box
        query = search_box.get()

        # Send a GET request to the server with the query
        r = requests.get(paths.bondjango_url + '/mouse/?search='+query, auth=(paths.bondjango_username, paths.bondjango_password))

        # Store the result in the search_object
        search_object.search_result = json.loads(r.text)

        # Close the tkinter window
        top.destroy()

    # Create a button widget in the window that calls the query_server function when clicked
    search_button = tk.Button(top, command=query_server, text='Search')
    search_button.pack(side=tk.BOTTOM)

    # Start the tkinter main loop
    top.mainloop()

    # Return the search_object
    return search_object


def get_models() -> List[str]:
    """
    This function retrieves the model names from the database schema.

    Returns:
    List[str]: A list of strings representing the model names in the database schema.
    """

    # Send a GET request to the server to retrieve the database schema
    r = requests.get(paths.bondjango_url, auth=(paths.bondjango_username, paths.bondjango_password))

    # Parse the response text as JSON and get the keys (model names)
    # Return the model names as a list of strings
    return list(json.loads(r.text).keys())


def search_main():
    """
    This function prints the first model name from the database schema.

    """
    print(get_models()[0])


def query_database(target_model: str, query: str = None) -> List[Dict[str, Union[str, int]]]:
    """
    This function queries the database for a specific model with an optional query string.

    Parameters:
    target_model (str): The model in the database to be queried.
    query (str, optional): The query string to filter the results. Defaults to None.

    Returns:
    List[Dict[str, Union[str, int]]]: A list of dictionaries representing the query results.

    Raises:
    AssertionError: If the 'next' field in the response data is not None, indicating that pagination is enabled.
    """
    if query is None:
        r = requests.get(paths.bondjango_url + '/' + target_model + '/from_python/',
                         auth=(paths.bondjango_username, paths.bondjango_password))
    else:
        parsed_query = fd.parse_search_string(query)
        url_string = '/?'
        for key in parsed_query.keys():
            if parsed_query[key] == 'ALL':
                continue
            if key in LOOKUPS:
                t_idx = parsed_query[key].find('T')
                formatted_time = parsed_query[key][t_idx:].replace('-', '%3A')
                formatted_date = parsed_query[key][:t_idx]
                url_field = LOOKUPS[key] + '=' + formatted_date + formatted_time + '&'
            else:
                url_field = key + '=' + parsed_query[key] + '&'
            url_string += url_field
        url_string = url_string[:-1]
        r = requests.get(paths.bondjango_url + '/' + target_model + '/from_python' + url_string,
                         auth=(paths.bondjango_username, paths.bondjango_password))
    data = json.loads(r.text)
    results = data['results']
    assert data['next'] is None, 'Pagination is on, revise serializer'
    return results


def update_entry(url: str, data: Dict[str, str]) -> requests.Response:
    """
    This function updates a database entry.

    Parameters:
    url (str): The URL of the database entry to be updated.
    data (Dict[str, str]): The data to be updated in the database entry.

    Returns:
    requests.Response: The response from the server after the update operation.
    """
    return requests.put(url, data=data, auth=(paths.bondjango_username, paths.bondjango_password))


def create_entry(url: str, data: Dict[str, Union[str, int]]) -> requests.Response:
    """
    This function creates a new database entry.

    Parameters:
    url (str): The URL where the new database entry will be created.
    data (Dict[str, Union[str, int]]): The data for the new database entry.

    Returns:
    requests.Response: The response from the server after the creation operation.
    """
    return requests.post(url, data=data, auth=(paths.bondjango_username, paths.bondjango_password))


def delete_multiple(target_model: str, query: str) -> List[requests.Response]:
    """
    This function deletes multiple entries in the database based on a query.

    Parameters:
    target_model (str): The model in the database where the entries will be deleted.
    query (str): The query to select the entries to be deleted.

    Returns:
    List[requests.Response]: A list of responses from the server after each delete operation.
    """
    # Query the database to get the entries to be deleted
    target_entries = query_database(target_model, query)

    delete_results = []
    # For each entry in the target entries
    for entry in target_entries:
        # Send a DELETE request to the server with the URL of the entry
        result = requests.delete(entry['url'], auth=(paths.bondjango_username, paths.bondjango_password))
        # Append the server response to the delete_results list
        delete_results.append(result)
    # Return the list of server responses
    return delete_results