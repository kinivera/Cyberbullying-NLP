def topn_n_words(df, nWords):
    #Top words used for each cyberbullying type
    for bully in df['cyberbullying_type'].unique():
        top_50_words=df.clean_data[df['cyberbullying_type']==bully].str.split(expand=True).stack().value_counts()[:nWords]
    print(top_50_words)

def wordCloud(df):
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    '''Word cloud is an image composed of words used in a particular text or subject, in which 
    the size of each word indicates its frequency or importance'''
    for bully in df.cyberbullying_type.unique():
        s=''
        for phrases in df.clean_data[df.cyberbullying_type==bully]:
            s=s+' '+phrases.strip()
            
        wordcloud=WordCloud(width=800,height=800,background_color='white',stopwords=STOPWORDS,min_font_size=10).generate(s)
        
        #plotting the word cloud image
        plt.figure(figsize=(8,8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        print()