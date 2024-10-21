from django import forms


class SentimentForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label='Enter your text')

