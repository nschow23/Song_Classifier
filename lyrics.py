


import os
import re

import lyricsgenius as lg
import numpy as np
import pandas as pd
import json

def get_lyrics(genius, arr, k, label):
    #####inputs
    ###genius object for API access
    ### arr : array containing list of artist names
    ### k: number of songs for each artist
    #####: label for the song: country = 1 , pop=0
    #### source : https://towardsdatascience.com/song-lyrics-genius-api-dcc2819c29
    c=0
    s=[]
    for name in arr:
        try:
            songs=(genius.search_artist(name,max_songs=k, sort='popularity')).songs
            for song in songs:
                lyrics = re.sub(r"[^A-Za-z\d'\s]+", '', song.lyrics)
                lyrics = re.sub(r"[0-9]", '', lyrics)
                lyrics = lyrics.replace("\n", " ")
                lyrics = lyrics.replace("EmbedShare URLCopyEmbedCopy", '')
                lyrics = lyrics.lower()
                lyrics= lyrics.replace('\u2005', ' ')
                s.append([name, lyrics, label])
        except:
            print(f"some exception at {name}: {c}")
    lyrics = np.array(s)
    print(lyrics.dtype)
    return lyrics

def get_lyrics_album(genius, arr, k, label):
    #####inputs
    ###genius abject for API access
    ### arr : array containing list of artist names
    ### k: number of songs for each artist
    #####: label for the song: country = 1 , pop=0
    c=0
    s=[]
    for name in arr:
        try:
            songs=(genius.search_artist(name,max_songs=k, sort='popularity')).songs
            for song in songs:
                lyrics = re.sub(r"[^A-Za-z\d'\s]+", '', song.lyrics)
                lyrics = re.sub(r"[0-9]", '', lyrics)
                lyrics = lyrics.replace("\n", " ")
                lyrics = lyrics.replace("EmbedShare URLCopyEmbedCopy", '')
                lyrics = lyrics.lower()
                lyrics= lyrics.replace('\u2005', ' ')
                s.append([name, lyrics, label])
        except:
            print(f"some exception at {name}: {c}")
    lyrics = np.array(s)
    print(lyrics.dtype)
    return lyrics

def main():
    print("Current working directory: {0}".format(os.getcwd()))
    Client_Access_Token = 'XrDjQeJyLfn9J3kLaWDIrsdIfzOYkTIAKCs2srMTGbfCePU2cs796yUNTzJLNfSb'
    genius = lg.Genius(Client_Access_Token, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
    artists = ['Logic', 'Morgan Wallen', 'Justin Bieber']
    country_artists = ['Morgan Wallen', 'Luke Combs', 'Kane Brown', 'Jason Aldean', 'Christ Stapleton', 'Luke Bryan',
                       'Carrie Underwood', 'Florida Georgia Line', 'Blake Shelton', 'Tyler Childers', 'Kenny Chesney',
                       'Miranda Lambert', 'Tim McGraw', 'Eric Church', 'Koe Wetzel', 'Cody Johnson', 'Toby Keith',
                       'Gabby Barrett', 'Keith Urban', 'Zac Brown Band']
    pop_artists = ['Justin Bieber','Lil Nas X', 'Olivia Rodrigo', 'Ed Sheeran', 'Doja Cat', 'The Weeknd', 'Adele',
                   'Jonas Brothers', 'Coldplay', 'Twenty One Pilots', 'Billie Eilish', 'Dua Lipa','Ariana Grande',
                   'Maroon 5', 'Lizzo', 'Bruno Mars', 'Camila Cabelo', 'Halsey', 'Demi Lovato', 'Drake', 'Post Malone']
    country_lyrics = get_lyrics(genius, country_artists, 10, 1)
    pop_lyrics = get_lyrics(genius, pop_artists, 10, 0)
    lyrics = np.append(country_lyrics, pop_lyrics, axis=0)
    np.savetxt('lyrics_plus.csv', lyrics, fmt=('%s'), delimiter=',')



if __name__ == '__main__':
    main()