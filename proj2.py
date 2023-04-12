import twitter
from functools import partial
from sys import maxsize as maxint
import sys
import time
from urllib.error import URLError
from http.client import BadStatusLine
import json
import twitter
import operator
import networkx as nx
import pylab as plt

def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.
    
    CONSUMER_KEY = 'Ct4ryOg3dNYOthzErmQNy65pB' 
    CONSUMER_SECRET = 'H9GqgWVUIzk5ZNbBA4YpWyBDoaSm40JtK8OjICSSZgRtKssYma'
    OAUTH_TOKEN = '1314294433211396096-ENLBxJ88y31oeUaRTlPCHk5bfHcbj3'
    OAUTH_TOKEN_SECRET = 'NbBzRDcmIwaYcN8MsZxbcYI0i5DXCQbqY539wkTG0D77a'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)
    
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api
#twitter cookbook #16
def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw): 
    
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
    
        if wait_period > 3600: # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e
    
        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes
    
        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429: 
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'\
                  .format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function
    
    wait_period = 2 
    error_count = 0 

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0 
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


# Nothing to see by displaying twitter_api except that it's now a
# defined variable
#twitter cookbook #19
def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
                              friends_limit=maxint, followers_limit=maxint):
    
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None), \
    "Must have screen_name or user_id, but not both"
    
    # See http://bit.ly/2GcjKJP and http://bit.ly/2rFz90N for details
    # on API parameters
    
    get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids, 
                              count=5000)
    get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids, 
                                count=5000)

    friends_ids, followers_ids = [], []
    
    for twitter_api_func, limit, ids, label in [
                    [get_friends_ids, friends_limit, friends_ids, "friends"], 
                    [get_followers_ids, followers_limit, followers_ids, "followers"]
                ]:
        
        if limit == 0: continue
        
        cursor = -1
        while cursor != 0:
        
            # Use make_twitter_request via the partially bound callable...
            if screen_name: 
                response = twitter_api_func(screen_name=screen_name, cursor=cursor)
            else: # user_id
                response = twitter_api_func(user_id=user_id, cursor=cursor)

            if response is not None:
                ids += response['ids']
                cursor = response['next_cursor']
        
            print('Fetched {0} total {1} ids for {2}'.format(len(ids),\
                  label, (user_id or screen_name)),file=sys.stderr)
        
            # XXX: You may want to store data during each iteration to provide an 
            # an additional layer of protection from exceptional circumstances
        
            if len(ids) >= limit or response is None:
                break

    # Do something useful with the IDs, like store them to disk...
    return friends_ids[:friends_limit], followers_ids[:followers_limit]

#my function based off slides
def getReciprocalFriends(id):
    friends_ids, followers_ids = get_friends_followers_ids(twitter_api,
                                                               user_id=id,
                                                               friends_limit=1000,
                                                               followers_limit=1000)
    recipFriends = set(friends_ids).intersection(set(followers_ids))
    #returns dict of users and followcount
    friendsFollowers = get_user_profile(twitter_api, user_ids=list(recipFriends))
    return top_5_users(friendsFollowers)

#modified from twitter cookbook #17
def get_user_profile(twitter_api, screen_names=None, user_ids=None):
   
    # Must have either screen_name or user_id (logical xor)
    assert (screen_names != None) != (user_ids != None), \
    "Must have screen_names or user_ids, but not both"
    
    items_to_info = {}

    items = screen_names or user_ids
    
    while len(items) > 0:

        # Process 100 items at a time per the API specifications for /users/lookup.
        # See http://bit.ly/2Gcjfzr for details.
        
        items_str = ','.join([str(item) for item in items[:100]])
        items = items[100:]

        if screen_names:
            response = make_twitter_request(twitter_api.users.lookup, 
                                            screen_name=items_str)
        else: # user_ids
            response = make_twitter_request(twitter_api.users.lookup, 
                                            user_id=items_str)
    
        for user_info in response:
            if screen_names:
                #changed this to have .get("followers_count")
                items_to_info[user_info['screen_name']] = user_info.get("followers_count")
            else: # user_ids
                items_to_info[user_info['id']] = user_info.get("followers_count")

    return items_to_info

#my function that is given a dictionary of users with their follower count
def top_5_users(dict):
    sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    top_5 = [x[0] for x in sorted_dict[:5]]
    return top_5

#some from twitter cookbook crawl followers
def crawl_followers(twitter_api, id, limit=100):
    
    # Resolve the ID for screen_name and start working with IDs for consistency 
    # in storage
    graph = nx.Graph()
    #initialize queue
    queue = getReciprocalFriends(id)

    graph.add_edges_from([(id, x) for x in queue])

    #Runs while the number at the end of while loop is less than 100
    while graph.number_of_nodes() < limit:
        #Takes each id from the queue
        for id in queue:
            #gets 5 most popular reciprocal friends from a given user id
            moreFriends = getReciprocalFriends(id)
            #Adds to graph
            graph.add_edges_from([(id, x) for x in moreFriends])
            #Adds the current id's most popular reciprocal friends to the end of the queue
            queue = queue + moreFriends
    return graph




# Sample usage
if __name__ == "__main__":
    twitter_api = oauth_login()


    #my friend curran (consented)
    graph = crawl_followers(twitter_api, "740351417890770944")
     #Draws graph
    nx.draw(graph)
    #Saves drawn graph to computer
    plt.savefig('graph.png')
    #Prints out the graph's number of nodes, diameter, and average distance
    f = open("output.txt", "w")
    f.write("Number of nodes = ", graph.number_of_nodes())
    f.write("\n")
    f.write("Diameter = ", nx.diameter(graph))
    f.write("\n")
    f.write("Average distance = ", nx.average_shortest_path_length(graph))
    f.close()