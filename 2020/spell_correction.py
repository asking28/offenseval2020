!pip install pyspellchecker
from spellchecker import SpellChecker

spell = SpellChecker(distance=1)

# find those words that may be misspelled
misspelled = spell.unknown(['tt'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))
def spell_check(df):
  for i in range(df.shape[0]):
    if i%100000==0:
      print("Reached {0}, percent {1}".format(i,float(i/df.shape[0])*100))
    words=df['text'][i].split()
    misspelled = spell.unknown(words)
    l=[]
    for word in words:
      if word in misspelled:
        word=spell.correction(word)
      else:
        word=word
      l.append(word)
    #words=[spell.correction(word) for word in words]
    df['text'][i]=' '.join(word for word in l)
  return df
 def spell_collection(df):
  spell_corrected = re.sub(r'(.)\1+', r'\1\1', df)
  return spell_corrected
 for file_name in list_df[50:]:
  #if file_name != 'data_0':
    #df_dummy=df_dummy.append(pd.read_csv(root_path+"/chunky/"+file_name),ignore_index=True)
  df_x=pd.read_csv(root_path+"/chunky/"+file_name)
  df_x['text']=df_x['text'].apply(spell_collection)
  print(file_name)
  df_x=spell_check(df_x)
  df_x.to_csv(root_path+"/chunky/"+file_name,index=False)