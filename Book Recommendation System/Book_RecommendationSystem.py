import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches 

book_data = pd.read_csv("books.csv")
rating_data = pd.read_csv("ratings.csv")
tag_data = pd.read_csv("book_tags.csv")
book_tags = pd.read_csv("tags.csv")
book_data = book_data.drop(columns=['id', 'best_book_id', 'work_id', 'isbn', 'isbn13', 'title', 'work_ratings_count',
                                    'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4',
                                    'ratings_5', 'image_url', 'small_image_url'])
book_data = book_data.dropna()
rating_data = rating_data.sort_values("user_id")
rating_data.drop_duplicates(subset=["user_id", "book_id"], keep=False, inplace=True)
book_data.drop_duplicates(subset='original_title', keep=False, inplace=True)
book_tags.drop_duplicates(subset='tag_id', keep=False, inplace=True)
tag_data.drop_duplicates(subset=['tag_id', 'goodreads_book_id'], keep=False, inplace=True)
book_data["Content"] = book_data['original_title'] + ' '\
                       + book_data['authors'] + ' ' + book_data['average_rating'].astype(str)
book_data = book_data.reset_index()
cv = CountVectorizer(stop_words='english')
tfidf_matrix = cv.fit_transform(book_data['Content'])
cosine_sim = cosine_similarity(tfidf_matrix)


def get_title_from_index(index):
    return book_data[book_data.index == index]['original_title'].values[0]


def get_index_from_title(title):
    return book_data[book_data.original_title == title]['index'].values[0]


def recommendation(book_name):
    book_index = get_index_from_title(book_name)
    similar_books = list(enumerate(cosine_sim[book_index]))
    sorted_list = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:]
    i = 0
    print("Top ten similar books are:")
    for elements in sorted_list:
        print(get_title_from_index(elements[0]))
        i = i+1
        if i > 10:
            break
        
        
book_list=book_data.original_title.to_list()
book_name = input("Enter Book Name: ")
closest = get_close_matches(book_name,book_list)
Book_name = closest[0]
recommendation(Book_name)
