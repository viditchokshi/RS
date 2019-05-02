import pandas as pd 
import numpy as np 

def prep():
    df = pd.read_csv(r'C:\Users\ASUS\Desktop\ML\ScrapedSets\electro.csv', encoding = 'utf-8')

    cust = pd.DataFrame(columns=['customerID',
    'customername',
    'customercountry',
    'customerstate',
    'customercity',
    'customerpin',
    'customerlat',
    'customerlong',
    'customertotalreview',
    'customerpositivereview',
    'customernegativereview'
    ])

    cust['customerID'] = df['web-scraper-order']
    cust = cust.reset_index()


    # Assigning the columns
    prod = pd.DataFrame(columns=['productID',
    'category',
    'subcategory',
    'name',
    'grade',
    'uom',
    'price',
    'brand',
    'attributes',
    'elements',
    'application',
    'industry',
    'packagesize',
    'currency',
    'warrenty',
    'gurante',
    'applicationindustry',
    'availablelocation',
    'producttype',
    'availabilitytype',
    'leadtime',
    'certification',
    'totalreview',
    'positivereview'
    ])

    # Assigning the columns
    prod['category'] = df['MultiLink']
    prod['subcategory'] = df['Category']
    prod['brand'] = df['Brand']
    prod['price'] = df['Price']
    prod['name'] = df['Product']
    prod['totalreview'] = df['ProductRating']

    prod = prod.reset_index()


    sup = pd.DataFrame(columns=['supplierID',
    'suppliername',
    'suppliercountry',
    'supplierstate',
    'suppliercity',
    'supplierpin',
    'supplierlat',
    'supplierlong',
    'suppliersellingarea',
    'suppliercertification',
    'suppliertotalreview',
    'supplierpositivereview',
    'suppliernegativereview'
    ])

    # Assigning the columns
    sup['suppliername'] = df['SellerName']
    sup['suppliertotalreview'] = df['FeedBack']

    sup = sup.reset_index()


    new_df = pd.merge(prod, sup, how='outer')


    subset = new_df[['name', 'category', 'brand', 'totalreview', 'suppliername','suppliertotalreview']]

    #print subset.brand.value_counts()






    subset = subset.replace({
        'totalreview': '[A-Za-z]',
        'suppliertotalreview': '[A-Za-z]'
    }, '', regex = True)
    subset.head()

    subset = subset.replace({
        'totalreview': '5'
    }, '', regex = True)
    subset.head()

    subset = subset.replace({
        'suppliertotalreview': ' 5'
    }, '', regex = True)
    subset.head()

    subset['totalreview'] = pd.to_numeric(subset['totalreview'])
    subset['suppliertotalreview'] = pd.to_numeric(subset['suppliertotalreview'])

    subset.isna().sum()

    subset = subset.dropna()

    subset.isna().sum()
    subset.shape

    '''pop= subset.sort_values('totalreview', ascending=False)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,4))

    plt.barh(pop['category'].head(15),pop['totalreview'].head(15), align='center',
            color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularity Rating")
    plt.ylabel("Category")
    plt.title("Trending/Popular Categories")

    '''
    subset = subset.drop_duplicates(['name', 'totalreview', 'brand', 'category', 'suppliername'])

    '''plt.scatter(subset['suppliertotalreview'], subset['totalreview'], alpha = 0.4)
    plt.ylabel("Product Ratings")
    plt.xlabel("Seller FeedBack")
    plt.show()
    '''
    return subset

mydata = prep()

def compute(subset = mydata):
    #Import TfIdfVectorizer from scikit-learn
    from sklearn.feature_extraction.text import TfidfVectorizer

    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(subset['name'])

    #Output the shape of tfidf_matrix
    tfidf_matrix.shape

    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cs = compute()

def get_recommendations(index=280, cosine_sim=cs, subset = mydata):
        # Get the index of the movie that matches the title
        idx = index

        # Get the pairwsie similarity scores of all items with that item
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the item based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar item
        sim_scores = sim_scores[1:21]

        # Get the item indices
        item_indices = [i[0] for i in sim_scores]

        # Return the top most similar item
        return subset[['name', 'totalreview', 'brand', 'category', 'suppliername','suppliertotalreview']].iloc[item_indices]

pdata = get_recommendations()

def pproc(pdata):
    result = pdata


    #result.suppliername.value_counts()

    #result.brand.value_counts()

    g = result.groupby('suppliername')
    pd.set_option('display.max_columns', None)
    '''for g.suppliername,suppliername_df in g:
                    print '---------------------------------------------------------------------------'
                    print suppliername_df
                    print '---------------------------------------------------------------------------'
            '''
    '''result.name.unique()
            
                result.suppliername.unique()
            
                result[result['suppliername']=='Cloudtail India'].head(10)
            
                result[result['suppliername']=='Cloudtail India'].mean()
            '''
    '''plt.scatter(result.brand, result.totalreview)
    plt.title("Brand vs Rating")
    plt.legend()
    plt.show()
    '''
    '''import seaborn as sns
    sns.set()

    sns.boxplot(result.suppliertotalreview, result.suppliername, palette='bwr')

    sns.distplot(result.suppliertotalreview, color = "blue")
    '''
    #grp = ['Voltas shoppe', 'Cloudtail India']
    #top sellers above poor sellers for same product
    sorted_result = pd.DataFrame(columns=result.columns) 
    for i in result.name.unique():
        unique_result_set = result[result.name == i]
        unique_result_set = unique_result_set.sort_values(['suppliertotalreview'], ascending=False)
        sorted_result = sorted_result.append(unique_result_set, ignore_index=True)
    return sorted_result

def get_index(subset, search_string):
    product_indexes = []
    for counter in range(len(subset)):
        hasAllWords = True
        for word in search_string.split(" "):
            if not word.lower() in subset.iloc[counter]['name'].lower().split(" "):
                hasAllWords = False
        if hasAllWords:
            product_indexes.append(counter)
    return product_indexes
