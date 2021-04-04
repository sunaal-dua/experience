# importing required libraries
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys
print("installing langdetect python package...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
print("Done!")
from langdetect import detect

def lang_detect(text):
    """
    Detects language of the text. 
    If a language is detected, returns the ISO 639-1 language code, else "unknown"
    
    text: text to detect language (string)
    """
    try:                                                          
      return detect(text)                                      
    except:                                                       
      return 'unknown'    