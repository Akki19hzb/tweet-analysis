import tweepy
import csv, sys
import os.path,time
# Fill the X's with the credentials obtained by
# following the above mentioned procedure.
consumer_key = "sSQP5PVBS7J9dYlNIlOJZUHEl"
consumer_secret = "5rec9aHRnSPXhVmkv3HAbnUKaFm5eeEPMyQmWvNotTEoUrQn6S"
access_key = "284617609-wVPtfT8fysUOMw04anf2dGAAI4puTFR0kSKqkhU0"
access_secret = "XKmniYsU7KX0AsZsNfGb8GaZZULCbkYMcI5ejCpRMuR3a"

class ReturnStatus:
        Successful=False
class Return:
    status=ReturnStatus()
    data=None
    exception=None
dict={}

# Function to extract tweets
def get_tweets(username):
    # Authorization to consumer key and consumer secret
    ret=Return()

    try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            # Access to user's access key and access secret
            auth.set_access_token(access_key, access_secret)
            # Calling api
            api = tweepy.API(auth)
            # 200 tweets to be extracted
            number_of_tweets = 200
            # tweets = api.user_timeline(screen_name=username,count= 200, page=2)
            tweets = api.search(q="@"+username,count=100)

            # Empty Array
            tmp = []

            # create array of tweet information: username,
            # tweet id, date/time, text
            # print(tweets)
            # exit()
            tweet_id=[]
            tweet_text=[]
            # tweets_for_csv = [tweet.text for tweet in tweets]  # CSV file created
            with open('%s_tweets4.csv' % username,'a') as f:
                writer = csv.writer(f)
                # if os.path.exists('%s_tweets1.csv' % username):
                #     writer.writerow(["id", "tweets"])
                for tweet in tweets:

                    if(tweet.in_reply_to_status_id is None):
                        if tweet.id not in dict:
                            dict[tweet.id] = tweet.id
                            print(tweet.id,": ", tweet.text)
                            tweet_id.append(tweet.id)
                            tweet_text.append(tweet.text)
                            try:
                                writer.writerow([tweet.id,tweet.text])
                            except:
                                print("Oops!"  ,sys.exc_info()[0],"occured.")


                f.close()

                # tweet_id.append(tweet.id)
                # tweet_text.append(tweet.text)
            ret.status=True
    except  Exception as e:
        ret.status=False
        ret.exception=e
    finally:
        return ret




# Driver code
if __name__ == '__main__':
    # Here goes the twitter handle for the user
    # whose tweets are to be extracted.
    # get_tweets("airtel_presence
    while True:
        ret=get_tweets('airtel_presence')
        print(ret.exception)
        time.sleep(100)