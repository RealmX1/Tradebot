import praw
import nltk
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
from fasttext import load_model

def remove_noise(tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

topic = 'leagueoflegends'
num_submissions = 5
to_scrape = True
comments = []
cleaned_comments = []
comments_file = 'comments.txt'
lemmatized = []
neg_label = '__label__0'
neutral_threshold = 0.60

# scraping data from reddit
reddit = praw.Reddit(
     client_id='6TT6NjHtczRrMg',
     client_secret='d_jNGgQ2v7RdN2r7P4h6tmvQRdwLQA',
     user_agent='mac:personal script (by u/Practical-Tourist733).'
)

if to_scrape:
    for submission in reddit.subreddit(topic).hot(limit=num_submissions):
        # print(submission.title)
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comments.append(comment.body)
    with open(comments_file, 'w') as filehandle:
        for comment in comments:
            filehandle.write('%s\n' % comment)
comments = []
with open(comments_file, 'r') as filehandle:
    for line in filehandle:
        comment = line[:-1]
        comments.append(comment)

# delete urls and usernames
for comment in comments:
    has_url = re.search('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', comment)
    if has_url:
        continue
    comment = re.sub('\/u\w+', '', comment)
    comment = comment.strip()
    if comment != '':
        cleaned_comments.append(comment)

classifier = load_model('sentiment140/model_tweet.bin')
labels = classifier.predict(cleaned_comments)
num_positive = 0
num_negative = 0
num_neutral = 0
for label, prob in zip(labels[0], labels[1]):
    if prob[0] <= neutral_threshold:
        num_neutral += 1
    elif label[0] == neg_label:
        num_negative += 1
    else:
        num_positive += 1
tot = num_positive + num_negative + num_neutral
print("positive", "negative", "neutral")
print(num_positive/tot, num_negative/tot, num_neutral/tot)
print(num_positive, num_negative, num_neutral)

# tokenizing & removing noise & normalizing
# for comment in cleaned_comments:
#     tokens = nltk.word_tokenize(comment)
#     if len(tokens) > 0:
#         lemmatized.append(remove_noise(tokens, stopwords.words('english')))

# print(lemmatized)
