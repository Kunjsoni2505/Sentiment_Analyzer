from django import forms


class SentimentForm(forms.Form):
     # text = forms.CharField(widget=forms.Textarea, label='Enter your text')
      text = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Enter your text here...'}), label="Enter Your Text", required=True)
