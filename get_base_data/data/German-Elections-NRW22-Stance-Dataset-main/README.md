# NRW22-Stance: Dataset for Continuous Multi-Target Stance Detection towards German Political Actors

## 👋 Welcome

We present a multi-target stance detection dataset consisting of tweets collected during the 2022 German state election in North Rhine-Westphalia. It is divided into one training dataset and eight subsequent testing datasets that do not overlap in time. These datasets contain tweets, replies, and quoted retweets annotated for the task of target-dependend stance detection.

**IMPORTANT:** We provide only tweet IDs and our annotations. The original tweet texts and additional tweet metadata are not part of the public dataset.

## 🏷 Annotation Procedure

This dataset was labeled by three annotators on different subsets using a pre-agreed scheme, and interrater reliability was measured at approximately 0.75 Krippendorff's alpha. We labeled every unique explicit entity mention per tweet into three classes:

* **against** - if the author clearly expresses their opinion against a mentioned politician or party,
* **in favor** - if the author clearly expresses their opinion in support of a mentioned politician or party,
* **neither** - else

An entity mention can appear in the text as a user handle, plain text, or hashtag. In total, **13 different target entities** are considered: six parties and their seven leading candidates (Die Linke had two leading candidates). To identify party mentions, we used not only party-affiliated Twitter accounts but also names and accounts of party-related persons who were not leading candidates, such as other party members. In such cases, the target entity is affiliated with the party name; however, we additionally provide the full name of the person as an additional association. For instance, when *Christian Lindner* was mentioned, we annotated a stance for the corresponding party *FDP*, but his name is also provided in the dataset as an additional attribute.

We provide text spans for every extracted target entity mention. A text span can be affiliated with one or more entities. Names of party coalitions, such as *Ampelkoalition*, are affiliated with multiple target entities, as are hashtags containing multiple party names, such as *#NieWiederCDUCSU*.

While all extracted mentions of the same politician or party from tweets are provided in the dataset, we annotated only one stance per unique target entity in each tweet. Thus, mentioning the *Die Linke* party in two different positions within the tweet did not lead to two stance annotations for the *Die Linke* party.

In the case of replies, we annotated only unambiguous entity mentions. An unambiguous mention is considered when:

* An entity is mentioned within the text body of a reply:

  > *@recipient* bla bla bla **entity** bla bla bla

* An entity is the recipient, and there is only one recipient:

  > ***@entity*** bla bla bla bla bla bla bla bla

When an entity is mentioned as one of multiple reply recipients, it is considered ambiguous. In this case, it is unclear to whom the reply is directed and which recipient is only added in the sense of carbon copy (CC in email). We exclude such entity mentions from the annotation process. However, we still provide the corresponding extracted spans for them in our dataset:

* > *@recipient1* ***@entity*** *@recipient3* bla bla bla bla bla bla bla bla


## 📊 Details and Statistics

### Dataset Sizes and Sampling Methods

The entire dataset is split into nine time intervals. The first, longer time interval is used to sample tweets for the training dataset. The subsequent eight weekly intervals are used for the testing datasets. In advance, we used the Twitter API to collect tweets related to the 2022 German state election in North Rhine-Westphalia. The total number of collected tweets (excluding retweets) during the full period from 2022-01-31 to 2022-05-15 was *434,216*. We then used a stance detection model from [Müller et al. 2022](https://link.springer.com/chapter/10.1007/978-3-031-15086-9_9) to pseudo-label these tweets. From the pseudo-labeled tweets, we sampled datasets for manual annotation using two methods:

* *random* - we took at most N random tweets for each entity and each pseudo-labelled stance. For the training dataset, we conducted multiple iterations with different values of N.
* *popular* - we took at most the top 50 of the most-retweeted tweets (with more then one retweet) for each entity and each pseudo-labelled stance.

After sampling and annotating the tweets, we retained only those with at least one unambiguous entity mention. The table below summarizes the final tweet counts per dataset.

| **dataset** | **interval** | **from\_date** | **to\_date** | **num\_labeled\_tweets** | **sampling\_method** |
| :--     | :-- | :--        | :--        | :--  | :--    |
| train   | 0   | 2022-01-31 | 2022-03-20 | 9672 | random |
| test\_1 | 1   | 2022-03-21 | 2022-03-27 | 986  | random |
| test\_2 | 2   | 2022-03-28 | 2022-04-03 | 1371 | random + popular |
| test\_3 | 3   | 2022-04-04 | 2022-04-10 | 747  | popular |
| test\_4 | 4   | 2022-04-11 | 2022-04-17 | 912  | popular |
| test\_5 | 5   | 2022-04-18 | 2022-04-24 | 1225 | popular |
| test\_6 | 6   | 2022-04-25 | 2022-05-01 | 1341 | popular |
| test\_7 | 7   | 2022-05-02 | 2022-05-08 | 1630 | popular |
| test\_8 | 8   | 2022-05-09 | 2022-05-15 | 1859 | popular |


### Target Entity Distribution

Every tweet in the dataset contains at least one target entity. The table below summarizes the number of unique target entity mentions based on the entity type. We distinguish between parties and their leading candidates.


| **target\_entity** | **train** | **test\_1** | **test\_2** | **test\_3** | **test\_4** | **test\_5** | **test\_6** | **test\_7** | **test\_8** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| *Parties* |
| AfD | 1261 | 123 | 138 | 102 | 115 | 157 | 149 | 197 | 192 |
| Bündnis 90/Die Grünen | 1456 | 193 | 209 | 136 | 135 | 176 | 239 | 218 | 295 |
| CDU | 1819 | 159 | 236 | 156 | 164 | 230 | 251 | 300 | 245 |
| Die Linke | 740 | 49 | 81 | 34 | 49 | 93 | 54 | 115 | 160 |
| FDP | 1922 | 222 | 290 | 83 | 124 | 181 | 209 | 232 | 244 |
| SPD | 1805 | 205 | 285 | 124 | 143 | 214 | 255 | 278 | 285 |
| *Leading Candidates* |
| Carolin Butterwegge (Die Linke) | 21 | 1 | 1 | 1 | 0 | 2 | 3 | 2 | 10 |
| Hendrik Wüst (CDU) | 1751 | 162 | 204 | 150 | 180 | 272 | 286 | 366 | 478 |
| Joachim Stamp (FDP) | 871 | 166 | 224 | 17 | 64 | 48 | 52 | 118 | 116 |
| Jules El-Khatib (Die Linke) | 1036 | 7 | 19 | 1 | 33 | 1 | 7 | 14 | 17 |
| Markus Wagner (AfD) | 50 | 2 | 6 | 6 | 2 | 5 | 3 | 22 | 25 |
| Mona Neubaur (Bündnis 90/Die Grünen) | 420 | 128 | 147 | 31 | 10 | 52 | 45 | 83 | 66 |
| Thomas Kutschaty (SPD) | 811 | 117 | 211 | 74 | 128 | 173 | 236 | 274 | 309 |


### Stances Distribution

Every tweet has at least one annotated stance. The table below summarizes the number of stances based on the target entity.

| **label** | **train** | **test\_1** | **test\_2** | **test\_3** | **test\_4** | **test\_5** | **test\_6** | **test\_7** | **test\_8** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| against | 7379 | 984 | 1189 | 580 | 549 | 731 | 814 | 900 | 963 |
| in favor | 2697 | 262 | 556 | 244 | 291 | 326 | 425 | 573 | 667 |
| neither | 3887 | 288 | 306 | 91 | 307 | 547 | 550 | 746 | 812 |


### Tweet Type Distribution

The table below summarizes the tweet counts by type.

| **type** | **train** | **test\_1** | **test\_2** | **test\_3** | **test\_4** | **test\_5** | **test\_6** | **test\_7** | **test\_8** |
| :--            | :--  | :-- | :-- | :-- | :-- | :-- | :--  | :--  | :--  |
| tweet          | 4467 | 341 | 633 | 530 | 604 | 911 | 1031 | 1261 | 1442 |
| reply          | 4577 | 564 | 613 | 119 | 226 | 204 | 194  | 199  | 256  |
| quoted retweet | 628  | 81  | 125 | 98  | 82  | 110 | 116  | 170  | 161  |


## 🧱 Dataset Entry Structure

Every dataset part is provided as a JSON file consisting of a list of JSON objects. The object fields are:

* `tweet_id` *(str)* - Tweet ID as string. **NOTE**: Converting it to integers in a [Pandas's DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) can lead to unexpected behavior where the last two digits are rounded to zeros, potentially corrupting the ID.
* `labels` *Dict[str, str]* - Mapping of target entity names to stance labels.
* `annotated_pos_start` *int* - The beginning of the text part used for extracting unambiguous entities as a character offset (see explanations at the beginning). For tweets, quoted retweets, and replies with a single recipient, this is always *0*. For replies with multiple recipients, it is the beginning of the text body.
* `spans` *List[Dict[str, Any]]* - Unambiguous entity mentions as text spans that were considered for annotation:
    * `start` *int* - Start position of the text span as a character offset.
    * `end` *int* - End position of the text span an a character offset.
    * `affiliations` *List[Dict[str, str|null]* - Affiliated target entities for the span:
        * `name` *str* - Target entity name.
        * `affiliation_type` *str* - Type of target entity: *party* or *person* (i.e. leading candidate).
        * `association` *str|null* - Full name of a party-affiliated person, excluding leading candidates.
* `spans_not_annotated` *List[Dict[str, Any]]* - Ambigious entity mentions.

```json
[
    {
        "tweet_id": "1506020058266759168",

        // Entity name -> stance
        "labels": {
            "Die Linke": "against"
        },

        // Starting position (character offset) from which the text was considered for annotation
        "annotated_pos_start": 21,

        // Spans considered for stance annotation
        "spans": [
            {
                "start": 202,
                "end": 207,
                "affiliations": [
                    {
                        "name": "Die Linke",
                        "affiliation_type": "party",
                        "association": null
                    },
                    ... // more entity affiliations for this span
                ],
                "text": "Linke"
            },
            ... // more affiliated spans in this tweet
        ],

        // Spans excluded from stance annotation (in reply-to prefix)
        "spans_not_annotated": [
            {
                "start": 0,
                "end": 9,
                "affiliations": [
                    {
                        "name": "Die Linke",
                        "affiliation_type": "party",
                        "association": null
                    }
                ],
                "text": "@dieLinke"
            }
        ]
    },
    ... // more tweets
]
```

## ✍️ Citation

```
@misc{sparta2025nrw,
    author = {Müller, Arthur and Riedl, Jasmin and Drews, Wiebke and Steup, Johannes and Neumeier, Andreas},
    title = {{NRW22-Stance: Dataset for Continuous Multi-Target Stance Detection towards German Political Actors}},
    year = {2025},
    howpublished = {\url{<https://github.com/UnibwSparta/German-Elections-NRW22-Stance-Dataset/>}},
}
```
