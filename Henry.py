import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


# Initialize dataset with a small seed set of labeled articles
texts = [
    "couple of weeks ago, a Republican friend informed me that Vice President Kamala Harris, now the likely Democratic Party presidential candidate, had taken to quoting Karl Marx. Needless to say, I was both confused and pretty excited about this. But my heart sank as soon as I discovered his source:   I barely needed to watch the video to know that the infamous liar Christopher Rufo was lying once again. There’s no need to take my word for it — just watch the video for yourself. Harris talks a lot about equity in these clips, but she never even comes close to saying, “From each according to his ability, to each according to his needs.” That Rufo added the line that this “should sound familiar” makes it hard not to read it as a deliberate attempt to sucker his audience into thinking those sounds actually came out of her mouth.  Rufo, to be fair, earnestly believes that when Democrats say “equity” they really do have something like Marxism in mind. Here’s how he put it a few years ago:  Critical race theorists, masters of language construction, realize that “neo-Marxism” would be a hard sell. Equity, on the other hand, sounds nonthreatening and is easily confused with the American principle of equality. Can we stop for a moment and just appreciate how kooky this conspiracy theory really is? It would be one thing to say that equity is a Marxist idea or that it was inspired by Marxism, but that’s not what Rufo is doing here. Rufo is claiming that critical race theorists actually want to do Marxism, but that they consciously decided that they won’t be able to talk anyone into it and that they needed to use their powers as “masters of language construction” (whatever that means) to come up with a clever code word for it. How does Rufo know this? Has he ever provided any direct evidence of this elaborate deception? Isn’t it legitimately disquieting that anyone treats a man who says things like this as a serious thinker?  Anyway, compare these four situations:  Equality of assistance: Everyone gets the same assistance in a capitalist system. So, for example, the government sends everyone a $500 check, but other than that, nothing about our economy is changed. Equality of opportunity: The government tries to ensure that capitalism proceeds on a level playing field by compensating people for disadvantages they face that are unrelated to market competition. So, for example, imagine that someone cheated the welfare system out of millions of dollars by pretending they were poor and used that money to start a business. That puts everyone else at a disadvantage, so the government could try to fix that by either taking away millions from the cheater or giving millions to everyone else. Equality of outcome: Everyone always has the exact same amount of money, no matter how hard they work or don’t work. “To each according to their needs”: Everyone gets enough assistance to cover their needs, which may be unequal, and people may still earn unequal wealth on top of that. So, for example, if you need $500 worth of medicine every month, the government gives you $500; if I only need $20 worth of medicine, the government only gives me $20. However, we both have to work for any additional money we want in addition to that, and the government lets us earn as much as we can. What’s really funny about all of this is that Rufo thinks Kamala Harris wants to do (4), but he is also daft and thinks that (4) is actually (3). In fact, however, Harris is arguing that instead of doing (1), we should do (2). The ridiculous thing about nongovernmental organization (NGO) “equity” speak is that it really just means “equal opportunity,” the phrase Democrats used to say when they were interested in being understood by the general public.  Kamala Harris is obviously not a Marxist. Neither, for that matter, are most people in the world of NGOs, which is why NGO folks rallied behind Hillary Clinton and then Elizabeth Warren against the socialist candidate Bernie Sanders. At best, Harris and the NGOers just want to do the old Democratic program of capitalism plus some handouts for various disadvantaged groups.  Rufo is conflating this with Marxism, first because he probably does not understand the difference between the four economic approaches I outlined above, and second because he is a cynic who constantly fearmongers about imminent Marxism so that he can sell books.  .",  # Example left-leaning text
    "Minnesota Governor and Vice-Presidential hopeful Tim Walz has an abnormal relationship with China.  “No matter how long I live, I will never be treated that well again…it was an excellent experience,” Walz said upon his return from the first of many trips there. Walz claims that his Chinese hosts lavished him with “more gifts than I could bring home.”   Walz and his supporters would like us to believe that his cozy ties to the Chinese Communist Party (CCP) are not unusual. In fact, he is “hawkish” and staunch critic of the CCP’s human rights abuses, his media allies say. The Chinese have an expression for the practice of providing cover for the communist regime while issuing the occasional muted critiques: it’s called “big help with a little bad mouth.”  As Breitbart News reported, Walz first visited China in 1989 under the auspices of a now-shuttered Harvard program called WorldTeach. Walz and became so enamored with the country that he and his wife honeymooned there after their marriage on June 4, 1994, the fifth anniversary of the Tiananmen Square massacre. Walz “wanted to have a [wedding anniversary] date he’ll always remember,” according to his wife.   The pro-democracy protest movement of students in Tiananmen Square ended in a blood bath during the night of June 3 to June 4, 1989. Between 1,500 to 4,000 demonstrators were killed and 10,000 wounded when the Chinese Communist Party ordered a military crackdown on the protesters, and the People’s Liberation Army rolled their tanks into the square and opened fire on the crowd. The students were demonstrating to demand more democracy and freedom of thought from the Chinese government. (Jacques Langevin/Getty Images)   The pro-democracy protesters watch as the Chinese communist regime’s tanks rolled into Tiananmen Square during the bloody military crackdown of the democracy demonstrations on June 3 to 4, 1989. (Jacques Langevin/Getty Images)   The pro-democracy protesters are seen on Changan Avenue during the CCP’s bloody military crackdown on June 4, 1989, in Tiananmen Square. (Peter Charlesworth/LightRocket via Getty Images)   The pro-democracy protesters are seen after the military crackdown in Tiananmen Square on the night of June 3 to June 4, 1989. (Jacques Langevin/Getty Images)   People transport a wounded woman during the military crackdown in Tiananmen Square on June 3 to 4, 1989. (David Turnley/Getty Images)   Family members try to comfort a grief-stricken mother who just learned that her son was killed during the military crackdown in Tiananmen Square on June 4, 1989. (David Turnley/Getty Images)   A Chinese man stands alone to block a line of tanks heading towards Tiananmen Square on June 5, 1989, one day after the Tiananmen Square massacre. This courageous unknown man became known to the world simply as “Tank Man.” (AP Photo/Jeff Widener)  The Walz newlyweds brought sixty students with them on their honeymoon, kicking off the first of many student exchange trips to the communist country. Walz and his wife set up a company, Education Travel Adventures Inc., to facilitate the trips to China. They ultimately ended up taking more than thirty trips there.   Walz’s last known trip to China was in 2015, but he continues to meet with communist officials and headline CCP-backed events. Walz’s deep and abiding ties to China—including multiple connections to CCP intelligence front organizations—present troubling national security questions about a potential Harris-Walz administration.  Here are Walz’s top seven connections to the CCP that demand explanation.  1. The CCP approved and even subsidized Walz’s student exchanges.  After his first trip to China in 1989, Walz returned to his teaching job in America and hung a “Chinese banner” in his school office. By 1993, Walz was taking American students on visits to China where he told his students to “downplay their American-ness.”  When asked about why he was so interested in China so early on, Walz stated “China was coming, and that’s the reason that I went.” According to U.S. national security expert John Schindler, “no American would be allowed to run academic exchanges for a couple of decades, on the CCP’s dime, without [Ministry of State Security] approval. It just wouldn’t happen.”  Shockingly, Chinese authorities reportedly covered “a large part of the cost” of the 1993 summer trip. The next year, Walz and the Chinese government jointly sponsored scholarships for American students to visit China. Between 1989 and 2003, Walz travelled with hundreds of students to China.  Did Walz or his travel company, Education Travel Adventures Inc., receive any money from the Chinese government? His public financial disclosures do not go back far enough to know.  2. A CCP diplomat and other CCP government officials attended Walz’s gubernatorial inauguration in January 2019.  A translation from a Chinese government source reveals that, “Acting Consul General Liu Jun congratulated Governor Waltz and expressed his expectation to strengthen cooperation with the new Minnesota government to jointly promote the friendly and cooperative relations between Minnesota and China.”  Why were CCP members at the inauguration of a Minnesota governor and would Chinese diplomats congratulate a sincere critic of China’s human rights abuses?  3. The CCP Diplomat left the Walz inauguration to meet with Walz cronies at Minnesota’s premier globalist non-governmental organization (NGO), Minnesota Global.  According the translated Chinese government press release, “Acting Consul General Liu congratulated [Global Minnesota] on the successful holding of the China Theme Year event and said that the Consulate General looks forward to continuing to strengthen communication with [Global Minnesota] in the new year to promote friendly cooperation between Minnesota and China.”  Global Minnesota is close with Walz and has sponsored at least one his foreign trips (to Finland). Last December, Walz awarded a Global Minnesota nominee for a business award.  Global Minnesota is affiliated with globalist entities like the United Nations and frequently invites Walz for speaking engagements (in 2020, 2021, and 2022).  4. Walz has close connections to a Twin Cities-based organization that houses an alleged secret CCP police station—one of only seven secret CCP police stations in the U.S.  In 2022, Minnesota Global partnered with group called the Chinese American Association of Minnesota (CAAM) to send delegations to China.  CAAM has been accused of housing a CCP intelligence agency “Service Center” (which is effectively a secret CCP police station) in Minnesota. The Daily Caller reported:  The Chinese Communist Party’s (CCP) United Front Work Department (UFWD) — which at least one U.S. government commission has characterized as a “Chinese intelligence service” — operates so-called “Overseas Chinese Service Centers” (OCSCs) that are housed within various U.S.-based nonprofits. OCSCs were ostensibly set up to promote Chinese culture and assist Chinese citizens living abroad, according to Chinese government records.  In April 2023, the Justice Department busted an alleged CCP Ministry of Public Security outpost, which the DOJ called a “secret police station” used to “monitor and intimidate dissidents” and other critics of Beijing.  Why has Walz failed to shut down this Overseas Chinese Service Center operating out of Minnesota?  5. Then-Congressman Walz praised a CCP-backed event that he attended with CCP diplomats in 2018.   According the Consulate General of the People’s Republic of China in Chicago, the event Walz attended was “The Greatest Spirit: Embrace China—Beautiful Sichuan” hosted at Minnesota University’s Northrop Theater.  The event was sponsored by the CCP’s All-China Federation of Returned Overseas Chinese (ACFROC). According to the Chinese government website:  US Congressman Waltz commented that 30 years ago, he celebrated Mid-Autumn Festival for the first time in Foshan, Guangdong. As Mid-Autumn Festival stands out in Chinese culture as a special day of family reunion, it was his pleasure to enjoy this performance with everyone together. China and the US have a solid tradition of cultural exchanges, and hopefully both countries can maintain this tradition and amicable relations.  6. Less than one year into his first gubernatorial term, Walz was an honored guest speaker at multiple CCP-backed influence operation events in 2019.  Ten months after his inauguration, Walz accepted a speaking gig from a CCP-backed event, joining the president of the Chinese People’s Association for Friendship with Foreign Countries (CPAFFC) on the short list of speakers.  Walz was invited to “speak about his experiences [in China] and Minnesota’s connections with China.” Walz also was a guest speaker at the U.S. China Peoples Friendship Association convention in 2019 (alongside CPAFFC President Li Xaolin).  The CPAFFC is effectively a CCP “United Front” cutout that is accused of “directly and malignly influencing” state and local politicians in the U.S., according to the State Department. The CCP’s United Front influence operation is specifically tasked with “co-opting and neutralizing threats to the party’s rule and spreading its influence and propaganda overseas.” Beijing view United Front operation “magic weapon” to advance CCP objectives around the world.  Why is Walz so cozy with obvious CCP intelligence operatives and are they paying him for these speeches?  7. Walz has a long history of making outlandishly pro-CCP comments.  Walz has said that “going [to China] was one of the best things I have ever done” and that if the Chinese “had the proper leadership, there are no limits to what they could accomplish.” He claims that his teaching position Macau Polytechnic University “helped develop his knowledge of China’s unique international status.”  In 2011, Walz said he developed “a great admiration for and a close connection with the Chinese people” after teaching there. He indirectly praised the CCP’s brutal police when he said that China had “almost no crime.”  Walz has also claimed the U.S. does not need to have an “adversarial relationship” with China and that “there was no anti-American feeling [in China] whatsoever.”  Notably, Walz’s gentle criticisms often ignore the aspect of the Chinese system responsible for the brutality: communism. And he recently compared socialism to “neighborliness.” Walz seems to view human rights abuses as events that all countries commit from time to time and move on.  On the twentieth anniversary of Tiananmen Square massacre, Walz said that “every nation has its dark periods that it must come to grips with.” And that “this Nation [the US] is no exception.” While this is true, Walz refers to events in American history that happened over a century ago with far less bloodshed.  Meanwhile, the CCP’s ongoing “break their lineage, break their roots” persecution of ethnic minorities in China is rightly characterized as a crime against humanity and even a genocide.  Why does Tim Walz seem to downplay the undeniable bloodshed of communism and socialism?  This is a developing story.."  # Example right-leaning text
]
labels = [0, 1]  # 0 for left-leaning, 1 for right-leaning
predictions = []  # Store predictions for plotting
pipeline = None

def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ')
        return text
    else:
        return ''

def train_model():
    global pipeline
    if len(texts) > 0:
        pipeline = make_pipeline(TfidfVectorizer(stop_words='english', max_features=1000), LogisticRegression())
        pipeline.fit(texts, labels)
        # Save the model and vectorizer
        joblib.dump(pipeline.named_steps['logisticregression'], 'political_bias_model.pkl')
        joblib.dump(pipeline.named_steps['tfidfvectorizer'], 'tfidf_vectorizer.pkl')

def analyze_text(url):
    global pipeline
    text = get_text_from_url(url)
    if text and pipeline:
        text_vector = pipeline.named_steps['tfidfvectorizer'].transform([text])
        prediction = pipeline.named_steps['logisticregression'].predict(text_vector)
        prediction_proba = pipeline.named_steps['logisticregression'].predict_proba(text_vector)[:, 1]
        return prediction[0], prediction_proba[0]
    else:
        return None, None

def plot_predictions(predictions):
    plt.clf()  # Clear the plot
    plt.plot(predictions, 'bo', markersize=8)
    plt.ylim(-0.1, 1.1)
    plt.axhline(y=0.5, color='r', linestyle='-')  # Midpoint line
    plt.axhline(y=0, color='g', linestyle='--')  # Far-left line
    plt.axhline(y=1, color='g', linestyle='--')  # Far-right line
    plt.title('Political Bias of Articles')
    plt.xlabel('Article Index')
    plt.ylabel('Bias Score (0 = Far Left, 1 = Far Right)')
    plt.pause(0.05)  # Pause to update the plot dynamically

def main():
    global texts, labels, predictions
    plt.ion()  # Turn on interactive mode for dynamic updating
    plt.figure(figsize=(10, 4))

    while True:
        url = input("Enter the URL (or 'exit' to quit): ")
        if url.lower() == 'exit':
            break

        # Analyze the text with the current model
        predicted_label, confidence = analyze_text(url)
        if predicted_label is not None:
            # Automatically add the new article to the dataset with the predicted label
            text = get_text_from_url(url)
            texts.append(text)
            labels.append(predicted_label)
            predictions.append(confidence)

            # Retrain the model with the expanded dataset
            train_model()

            print(f"The predicted political leaning score for the URL is: {confidence:.2f}")
            print(f"Label automatically assigned: {'Right-Leaning' if predicted_label == 1 else 'Left-Leaning'}")

            # Plot the updated predictions
            plot_predictions(predictions)
        else:
            print("Failed to extract or analyze text.")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open at the end of the program

if __name__ == "__main__":
    train_model()  # Train the model initially with the seed data
    main()
