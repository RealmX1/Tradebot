import tweepy

auth = tweepy.OAuthHandler("3E0U5EezZmXnyrsJrUXOtCfqM", "40ywqXVdhSlhGYz0N48pum9GBtJoRVPgRXLsDFeVHcbgNtmEwC")
auth.set_access_token("1400153836745019394-tYFOtNawvVC0hRxJRQ1CS3R5rQ4zZF", "15opbNuh3vJ8kJbbieNTc496sK68NdUI10187m7UDXcPA")

api = tweepy.API(auth)

public_tweets = api.home_timeline()
cursor = tweepy.Cursor(api.user_timeline, id = "twitter")
for tweet in cursor.items(10):
    print(tweet.text)