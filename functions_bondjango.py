import requests
import paths
import json
import tkinter as tk
from collections import defaultdict
import functions_data_handling as fd


# define the dictionary between query characters and lookups
LOOKUPS = {
    'gtdate': 'date__gt',
    'ltdate': 'date__lt',
    'slug': 'slug__iexact',
    'notes': 'notes__icontains',
}


class SearchBon:
    """Search object class"""
    def __init__(self, search_result):
        self.search_result = search_result


def parse_results(searchbon_object):
    """Parse search results into a dict"""
    parsed_results = defaultdict(list)
    for index in searchbon_object.search_result['results']:
        for k, v in index.items():
            parsed_results[k].append(v)
    return parsed_results


def query_gui():
    """Run a small GUI to deliver a search query to the database"""
    # create the window
    top = tk.Tk()
    # create the search box
    search_box = tk.Entry(top)
    search_box.pack(side=tk.TOP)
    # initialize the searchbon object
    search_object = SearchBon

    def query_server():
        """Query the server"""
        # get the text from the entry field
        query = search_box.get()
        # make the request to the server
        r = requests.get(paths.bondjango_url + '/mouse/?search='+query, auth=(paths.bondjango_username, paths.bondjango_password))
        # decode the json output from the server and store in the SearchBon object
        search_object.search_result = json.loads(r.text)
        # close the window
        top.destroy()
    # create the search button
    search_button = tk.Button(top, command=query_server, text='Search')
    search_button.pack(side=tk.BOTTOM)
    # start GUI loop
    top.mainloop()
    # return the search object created
    return search_object


def get_models():
    """get the model names from the database schema"""
    r = requests.get(paths.bondjango_url, auth=(paths.bondjango_username, paths.bondjango_password))
    return list(json.loads(r.text).keys())


def search_main():
    print(get_models()[0])
    # return parse_results(query_gui())


def query_database(target_model, query=None):
    """Same as query server but more general"""
    # if there is no search query, query all the instances
    if query is None:
        r = requests.get(paths.bondjango_url + '/' + target_model + '/from_python/',
                         auth=(paths.bondjango_username, paths.bondjango_password))
    else:
        # parse the query
        parsed_query = fd.parse_search_string(query)
        # initialize the url
        url_string = '/?'
        # for all the fields
        for key in parsed_query.keys():
            # if the item is ALL, skip since it's not relevant for the search
            if parsed_query[key] == 'ALL':
                continue
            if key in LOOKUPS:
                # find the T
                t_idx = parsed_query[key].find('T')
                formatted_time = parsed_query[key][t_idx:].replace('-', '%3A')
                formatted_date = parsed_query[key][:t_idx]
                url_field = LOOKUPS[key] + '=' + formatted_date + formatted_time + '&'
            else:
                url_field = key + '=' + parsed_query[key] + '&'
            # add it to the url
            url_string += url_field
        # remove the last ampersand as it is extra
        url_string = url_string[:-1]

        # make the request to the server
        r = requests.get(paths.bondjango_url + '/' + target_model + '/from_python' + url_string,
                         auth=(paths.bondjango_username, paths.bondjango_password))
    # read the data
    data = json.loads(r.text)
    # get the results
    results = data['results']

    assert data['next'] is None, 'Pagination is on, revise serializer'

    return results


def update_entry(url, data):
    """Update a database entry"""
    return requests.put(url, data=data, auth=(paths.bondjango_username, paths.bondjango_password))


def create_entry(url, data):
    """Create a database entry"""
    return requests.post(url, data=data, auth=(paths.bondjango_username, paths.bondjango_password))


