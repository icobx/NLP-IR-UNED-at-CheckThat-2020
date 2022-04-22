# -*- coding: utf-8 -*-
from os.path import join, dirname
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')


# --- token and char count ---

def char_count(text_string):
    return len(text_string)


def word_count(text_tokens):
    return len(text_tokens)


def average_word_count(text_string, text_tokens):
    if char_count(text_string) > 0:
        return char_count(text_string) / word_count(text_tokens)
    else:
        return 0


def n_letter_word_count(text_tokens, n):
    return sum(len(token) > n for token in text_tokens)


def type_token_ratio(text_tokens):
    if word_count(text_tokens) > 0:
        return word_count(unique(text_tokens)) / word_count(text_tokens)
    else:
        return 0


# --- pos count ---

def noun_count(text_tags):
    return text_tags.count("NN") + \
        text_tags.count("NNS") + \
        text_tags.count("NNP") + \
        text_tags.count("NNPS")


def main_verb_count(text_tags):
    return text_tags.count("VB") + \
        text_tags.count("VBD") + \
        text_tags.count("VBG") + \
        text_tags.count("VBN") + \
        text_tags.count("VBP") + \
        text_tags.count("VBZ")


def adjective_count(text_tags):
    return text_tags.count("JJ") + \
        text_tags.count("JJR") + \
        text_tags.count("JJS")


def adverb_count(text_tags):
    return text_tags.count("RB") + \
        text_tags.count("RBR") + \
        text_tags.count("RBS") + \
        text_tags.count("WRB")


def modal_verb_count(text_tags):
    return text_tags.count("MD")


def verb_count(text_tags):
    return main_verb_count(text_tags) + \
        modal_verb_count(text_tags)


def modifiers_count(text_tags):
    return adjective_count(text_tags) + \
        adverb_count(text_tags)


def is_content_word(pos_tag):
    return pos_tag in ("NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                       "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB")


def content_word_diversity(text_tokens, text_tags):
    content_word_list = []

    for i in range(len(text_tokens)):
        if is_content_word(text_tags[i]):
            content_word_list.append(text_tokens[i])

    unique_content_word_count = word_count(unique(content_word_list))

    content_word_count = word_count(content_word_list)

    if content_word_count > 0:
        return unique_content_word_count / content_word_count
    else:
        return 0


def content_tag_diversity(text_tags):
    unique_content_tag_count = noun_count(unique(text_tags)) + \
        main_verb_count(unique(text_tags)) + \
        modifiers_count(unique(text_tags))

    content_tag_count = noun_count(text_tags) + main_verb_count(text_tags) + modifiers_count(text_tags)

    if content_tag_count > 0:
        return unique_content_tag_count / content_tag_count
    else:
        return 0


# --- cue words count ---

def causation_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("thus") + \
        text_tokens.count("such") + \
        text_tokens.count("therefore") + \
        text_tokens.count("consequently") + \
        text_tokens.count("result") + \
        text_tokens.count("hence") + \
        text_tokens.count("accordingly") + \
        text_tokens.count("since") + \
        text_tokens.count("as") + \
        text_tokens.count("because") + \
        text_tokens.count("due") + \
        text_tokens.count("owing") + \
        text_tokens.count("effect")


def exclusive_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("except") + \
        text_tokens.count("apart") + \
        text_tokens.count("but") + \
        text_tokens.count("without") + \
        text_tokens.count("else") + \
        text_tokens.count("otherwise")


def negation_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("not") + \
        text_tokens.count("n't") + \
        text_tokens.count("never") + \
        text_tokens.count("no")


def negative_emotions_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("abominable") + \
        text_tokens.count("aching") + \
        text_tokens.count("afflicted") + \
        text_tokens.count("afraid") + \
        text_tokens.count("aggressive") + \
        text_tokens.count("agonized") + \
        text_tokens.count("alarmed") + \
        text_tokens.count("alienated") + \
        text_tokens.count("alone") + \
        text_tokens.count("angry") + \
        text_tokens.count("anguish") + \
        text_tokens.count("annoyed") + \
        text_tokens.count("anxious") + \
        text_tokens.count("appalled") + \
        text_tokens.count("ashamed") + \
        text_tokens.count("abnormal") + \
        text_tokens.count("bad") + \
        text_tokens.count("bitter") + \
        text_tokens.count("boiling") + \
        text_tokens.count("bored") + \
        text_tokens.count("betrayed") + \
        text_tokens.count("cold") + \
        text_tokens.count("cowardly") + \
        text_tokens.count("cross") + \
        text_tokens.count("crushed") + \
        text_tokens.count("complaining") + \
        text_tokens.count("cheated") + \
        text_tokens.count("confused") + \
        text_tokens.count("crappy") + \
        text_tokens.count("dejected") + \
        text_tokens.count("depressed") + \
        text_tokens.count("deprived") + \
        text_tokens.count("desolate") + \
        text_tokens.count("despair") + \
        text_tokens.count("desperate") + \
        text_tokens.count("despicable") + \
        text_tokens.count("detestable") + \
        text_tokens.count("diminished") + \
        text_tokens.count("disappointed") + \
        text_tokens.count("discouraged") + \
        text_tokens.count("disgusting") + \
        text_tokens.count("disillusioned") + \
        text_tokens.count("disinterested") + \
        text_tokens.count("dismayed") + \
        text_tokens.count("dissatisfied") + \
        text_tokens.count("distressed") + \
        text_tokens.count("distrustful") + \
        text_tokens.count("dominated") + \
        text_tokens.count("doubtful") + \
        text_tokens.count("doubtful") + \
        text_tokens.count("dull") + \
        text_tokens.count("embarrassed") + \
        text_tokens.count("empty") + \
        text_tokens.count("enraged") + \
        text_tokens.count("evil") + \
        text_tokens.count("excluded") + \
        text_tokens.count("exiled") + \
        text_tokens.count("fatigued") + \
        text_tokens.count("fearful") + \
        text_tokens.count("forced") + \
        text_tokens.count("frightened") + \
        text_tokens.count("frustrated") + \
        text_tokens.count("fuming") + \
        text_tokens.count("grief") + \
        text_tokens.count("grieved") + \
        text_tokens.count("guilty") + \
        text_tokens.count("hateful") + \
        text_tokens.count("heartbroken") + \
        text_tokens.count("helpless") + \
        text_tokens.count("hesitant") + \
        text_tokens.count("hesitant") + \
        text_tokens.count("hostile") + \
        text_tokens.count("humiliated") + \
        text_tokens.count("hurt") + \
        text_tokens.count("incapable") + \
        text_tokens.count("incensed") + \
        text_tokens.count("indecisive") + \
        text_tokens.count("indifferent") + \
        text_tokens.count("indignant") + \
        text_tokens.count("inferior") + \
        text_tokens.count("inflamed") + \
        text_tokens.count("infuriated") + \
        text_tokens.count("injured") + \
        text_tokens.count("insensitive") + \
        text_tokens.count("insulting") + \
        text_tokens.count("irritated") + \
        text_tokens.count("lifeless") + \
        text_tokens.count("lonely") + \
        text_tokens.count("lost") + \
        text_tokens.count("lousy") + \
        text_tokens.count("liar") + \
        text_tokens.count("lame") + \
        text_tokens.count("livid") + \
        text_tokens.count("menaced") + \
        text_tokens.count("miserable") + \
        text_tokens.count("misgiving") + \
        text_tokens.count("mournful") + \
        text_tokens.count("misunderstood") + \
        text_tokens.count("manipulated") + \
        text_tokens.count("nervous") + \
        text_tokens.count("neutral") + \
        text_tokens.count("nonchalant") + \
        text_tokens.count("negated") + \
        text_tokens.count("offended") + \
        text_tokens.count("offensive") + \
        text_tokens.count("objected") + \
        text_tokens.count("overwhelmed") + \
        text_tokens.count("obstructed") + \
        text_tokens.count("pained") + \
        text_tokens.count("pained") + \
        text_tokens.count("panic") + \
        text_tokens.count("paralyzed") + \
        text_tokens.count("pathetic") + \
        text_tokens.count("perplexed") + \
        text_tokens.count("pessimistic") + \
        text_tokens.count("pessimistic") + \
        text_tokens.count("powerless") + \
        text_tokens.count("preoccupied") + \
        text_tokens.count("provoked") + \
        text_tokens.count("quaking") + \
        text_tokens.count("questioned") + \
        text_tokens.count("rejected") + \
        text_tokens.count("repugnant") + \
        text_tokens.count("resentful") + \
        text_tokens.count("reserved") + \
        text_tokens.count("restless") + \
        text_tokens.count("sad") + \
        text_tokens.count("scared") + \
        text_tokens.count("shaky") + \
        text_tokens.count("shy") + \
        text_tokens.count("skeptical") + \
        text_tokens.count("sore") + \
        text_tokens.count("sorrowful") + \
        text_tokens.count("stupefied") + \
        text_tokens.count("sulky") + \
        text_tokens.count("suspicious") + \
        text_tokens.count("tearful") + \
        text_tokens.count("tense") + \
        text_tokens.count("terrible") + \
        text_tokens.count("terrified") + \
        text_tokens.count("threatened") + \
        text_tokens.count("timid") + \
        text_tokens.count("tormented") + \
        text_tokens.count("tortured") + \
        text_tokens.count("tragic") + \
        text_tokens.count("unbelieving") + \
        text_tokens.count("uncertain") + \
        text_tokens.count("uneasy") + \
        text_tokens.count("unhappy") + \
        text_tokens.count("unpleasant") + \
        text_tokens.count("unsure") + \
        text_tokens.count("upset") + \
        text_tokens.count("useless") + \
        text_tokens.count("unloved") + \
        text_tokens.count("unimportant") + \
        text_tokens.count("unconnected") + \
        text_tokens.count("victimized")


def cognitive_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("reason") + \
        text_tokens.count("deliberate") + \
        text_tokens.count("ideate") + \
        text_tokens.count("muse") + \
        text_tokens.count("ponder") + \
        text_tokens.count("consider") + \
        text_tokens.count("contemplate") + \
        text_tokens.count("deliberate") + \
        text_tokens.count("study") + \
        text_tokens.count("reflect") + \
        text_tokens.count("imagine") + \
        text_tokens.count("conceive") + \
        text_tokens.count("examine") + \
        text_tokens.count("estimate") + \
        text_tokens.count("evaluate") + \
        text_tokens.count("appraise") + \
        text_tokens.count("resolve") + \
        text_tokens.count("ruminate") + \
        text_tokens.count("scan") + \
        text_tokens.count("confer") + \
        text_tokens.count("consult") + \
        text_tokens.count("meditate") + \
        text_tokens.count("speculate") + \
        text_tokens.count("deem") + \
        text_tokens.count("hold") + \
        text_tokens.count("imagine") + \
        text_tokens.count("guess") + \
        text_tokens.count("presume") + \
        text_tokens.count("conceive") + \
        text_tokens.count("invent") + \
        text_tokens.count("create") + \
        text_tokens.count("recollect") + \
        text_tokens.count("recall") + \
        text_tokens.count("reminisce")


def first_person_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("i") + \
        text_tokens.count("we") + \
        text_tokens.count("me") + \
        text_tokens.count("us") + \
        text_tokens.count("my") + \
        text_tokens.count("our") + \
        text_tokens.count("mine") + \
        text_tokens.count("ours")


def second_person_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("you") + \
        text_tokens.count("your") + \
        text_tokens.count("yours")


def third_person_word_count(text_tokens):
    text_tokens = lowercase_list(text_tokens)

    return text_tokens.count("he") + \
        text_tokens.count("she") + \
        text_tokens.count("it") + \
        text_tokens.count("they") + \
        text_tokens.count("him") + \
        text_tokens.count("her") + \
        text_tokens.count("them") + \
        text_tokens.count("his") + \
        text_tokens.count("its") + \
        text_tokens.count("their") + \
        text_tokens.count("hers") + \
        text_tokens.count("theirs")


# --- auxiliary functions ---

def unique(text_tokens):
    return list(set(text_tokens))


def lowercase_list(text_tokens):
    return [token.lower() for token in text_tokens]


def get_feature_matrix(data_series, size):
    char_count_column = np.zeros(size)
    word_count_column = np.zeros(size)
    type_token_ratio_column = np.zeros(size)
    average_word_count_column = np.zeros(size)
    n_letter_word_count_column = np.zeros(size)

    noun_count_column = np.zeros(size)
    verb_count_column = np.zeros(size)
    main_verb_count_column = np.zeros(size)
    modal_verb_count_column = np.zeros(size)
    modifiers_count_column = np.zeros(size)
    adjectives_count_column = np.zeros(size)
    adverbs_count_column = np.zeros(size)
    content_word_diversity_column = np.zeros(size)
    content_tag_diversity_column = np.zeros(size)

    causation_word_count_column = np.zeros(size)
    exclusive_word_count_column = np.zeros(size)
    negation_word_count_column = np.zeros(size)
    negative_emotions_word_count_column = np.zeros(size)
    cognitive_word_count_column = np.zeros(size)
    first_person_word_count_column = np.zeros(size)
    second_person_word_count_column = np.zeros(size)
    third_person_word_count_column = np.zeros(size)

    text_id = 0
    for text in data_series:
        tokenized_text = word_tokenize(text)
        text_pos = pos_tag(tokenized_text)
        text_tags = [b for (a, b) in text_pos]

        char_count_column[text_id] = char_count(text)
        word_count_column[text_id] = word_count(tokenized_text)
        type_token_ratio_column[text_id] = type_token_ratio(tokenized_text)
        average_word_count_column[text_id] = average_word_count(text, tokenized_text)
        n_letter_word_count_column[text_id] = n_letter_word_count(tokenized_text, 6)

        noun_count_column[text_id] = noun_count(text_tags)
        verb_count_column[text_id] = verb_count(text_tags)
        main_verb_count_column[text_id] = main_verb_count(text_tags)
        modal_verb_count_column[text_id] = modal_verb_count(text_tags)
        modifiers_count_column[text_id] = modifiers_count(text_tags)
        adjectives_count_column[text_id] = adjective_count(text_tags)
        adverbs_count_column[text_id] = adverb_count(text_tags)
        content_word_diversity_column[text_id] = content_word_diversity(tokenized_text, text_tags)
        content_tag_diversity_column[text_id] = content_tag_diversity(text_tags)

        causation_word_count_column[text_id] = causation_word_count(tokenized_text)
        exclusive_word_count_column[text_id] = exclusive_word_count(tokenized_text)
        negation_word_count_column[text_id] = negation_word_count(tokenized_text)
        negative_emotions_word_count_column[text_id] = negative_emotions_word_count(tokenized_text)
        cognitive_word_count_column[text_id] = cognitive_word_count(tokenized_text)
        first_person_word_count_column[text_id] = first_person_word_count(tokenized_text)
        second_person_word_count_column[text_id] = second_person_word_count(tokenized_text)
        third_person_word_count_column[text_id] = third_person_word_count(tokenized_text)

        text_id = text_id + 1

    feature_matrix = np.array([
        char_count_column,
        word_count_column,
        noun_count_column,
        verb_count_column,
        main_verb_count_column,
        modal_verb_count_column,
        modifiers_count_column,
        adjectives_count_column,
        adverbs_count_column,
        content_word_diversity_column,
        content_tag_diversity_column,
        type_token_ratio_column,
        average_word_count_column,
        n_letter_word_count_column,
        causation_word_count_column,
        exclusive_word_count_column,
        negation_word_count_column,
        negative_emotions_word_count_column,
        cognitive_word_count_column,
        first_person_word_count_column,
        second_person_word_count_column,
        third_person_word_count_column
    ])

    feature_matrix = feature_matrix.transpose()
    feature_csr_matrix = csr_matrix(feature_matrix)

    return feature_csr_matrix


def get_pos_df(df):
    df.loc[:, "text_pos"] = pd.Series("", index=df.index)
    df = df[["line_number", "speaker", "text", "text_pos", "label"]].reset_index(drop=True)

    for i in range(len(df.index)):
        tokenized_text = word_tokenize(df.at[i, "text"])
        text_pos = pos_tag(tokenized_text)
        text_tags = [b for (a, b) in text_pos]
        df.at[i, "text_pos"] = " ".join(text_tags)

    return df


def show_feature_importance(feature_matrix, labels, limit):

    column_names = ["char_count", "word_count", "noun_count", "verb_count", "main_verb_count",
                    "modal_verb_count", "modifiers_count", "adjectives_count", "adverbs_count",
                    "content_word_diversity", "content_tag_diversity", "type_token_ratio",
                    "average_word_count", "n_letter_word_count", "causation_word_count",
                    "exclusive_word_count", "negation_word_count", "negative_emotions_word_count",
                    "cognitive_word_count", "first_person_word_count", "second_person_word_count",
                    "third_person_word_count"]

    limit = len(column_names)

    feature_df = pd.DataFrame(feature_matrix.toarray())
    feature_df.columns = column_names

    best_features = SelectKBest(score_func=chi2, k=limit)
    fitted_features = best_features.fit(feature_df, labels)
    scores_df = pd.DataFrame(fitted_features.scores_)
    columns_df = pd.DataFrame(feature_df.columns)

    feature_scores = pd.concat([columns_df, scores_df], axis=1)
    feature_scores.columns = ["Feature", "Score"]
    print(feature_scores.nlargest(limit, "Score"))
