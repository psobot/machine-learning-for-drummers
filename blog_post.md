# Machine Learning for Drummers

## Peter Sobot, June 24, 2018

First and foremost, I'm a drummer. At my day job, I work on machine learning
systems for recommending music to people at [Spotify](https://spotify.com).
But outside my 9-to-5, I'm a musician, and my journey through music started as
a drummer. When I'm not drumming, I'll often be creating electronic music -
with a lot of drums in it, of course.

If you're not familiar with electronic music production, many (if not most)
modern electronic music uses _drum samples_ rather than real, live recordings
of drummers to provide the rhythm. These drum samples are often distributed
professionally, as sample packs, or created by musicians and shared for free
online. Often, though, these samples can be hard to use, as their labeling and
classification leaves a lot to be desired:

// Screenshot of my sample folder with thousands of unlabeled files

Various companies have tried to tackle this problem by creating their own
proprietary formats for sample packs, such as Native Instruments' _Battery_ or
_Kontakt_ formats. However, these are all (usually) expensive software
packages and require you to learn their workflows.

In an effort to better understand how to use machine learning techniques, I
decided to use machine learning to try to solve this fairly simple problem:

> Is a given audio file a sample of a kick drum, snare drum, hi-hat, other
> percussion, or something else?

In machine learning, this is often called a [classification
problem](https://en.wikipedia.org/wiki/Statistical_classification), because it
takes some data and _classifies_ (as in _chooses a class for_) it. You might
think of this as a kind of **automated sorting system** (although I'm using
the word "sorting" here to mean "sort into groups" rather than "to put in a
specific ranking or order").

For those unfamiliar with machine learning, you might say:

> Why not just train the computer to learn what a kick drum is (and so on) by
> giving it a whole bunch of data?

This is _mostly_ correct already! (Hooray, you're a machine learning
engineer!)

The trouble comes from deciding what _data_ means in the above sentence. We
could:

 1. Give the computer all of the data we have and let "_machine learning_"
    figure out what's important and what's not
 1. Give the computer all of the data we have, but do a bit of pre-processing
    first to hint at parts of the data that might be important, then have
    "_machine learning_" classify our samples for us

Option 1 above is tricky, as our data comes in many different forms - long
audio files, short audio files, different formats, different bit depths,
sample rates, and so on. Throwing all of this at a machine and asking it to
make sense of it would require a _lot_ of data for it to figure out what we
humans already know.

Instead of making the computer do a ton of extra work, we can use option 2 as
a middle ground: we can choose some _things_ about the audio samples that we
think might be relevant to the problem, and provide those _things_ to a
machine learning algorithm and have it do the math for us. These _things_ are
known as
[_features_](https://en.wikipedia.org/wiki/Feature_(machine_learning)).

(If this word is confusing, think of a feature just like a feature of, say, a
TV - only instead of "42-inch screen" and "HDMI input", our features might be
"4.2 seconds long" and "maximum loudness 12dB". The word means the same thing
in both contexts.)

This process of figuring out what features we want to use is commonly known
as _feature extraction_, which makes sense. Given our input data (audio files),
let's come up with a list of features that us, as humans, might find relevant
to deciding if the file is a kick drum or a snare drum.

 - **Overall file length** is one simple feature - it's easy to measure, and it's
   possible that maybe a snare drum's sound continues on for longer than a
   kick drum's sound. (To prevent us from getting false positives here, let's
   only count the length of time that the sound is not silent, or **not quieter
   than -60dB**, in the file.)

 - **Overall loudness** might sound like a great feature to use (as maybe kicks
   are louder than snares?) but most samples used in electronic music are
   [_normalized_](https://en.wikipedia.org/wiki/Audio_normalization), meaning
   their loudness is adjusted to be consistent between files. Instead, we can
   use **maximum loudness**, **minimum loudness**, and **loudness at middle**
   (that is, loudness at the 50% mark through the file) to get a better idea
   for how the loudness changes over time. Drum hits should be loudest at the
   start of the sample, and should quickly taper off to silence.

 - Humans can tell the difference between kick drums and snare drums
   intuitively, and we do so by listening to the frequencies present in the
   sound. Kick drum samples have a lot more low-frequency content in them,
   as kick drums sound low and bassy due to their large diameter. To teach
   this to a machine learning algorithm, we can take the **average loudness in
   several frequency ranges** to tell the algorithm a little more about the
   [timbre of the sound](https://en.wikipedia.org/wiki/Timbre) as humans might
   hear it. (To better represent how this changes over time, we might take
   this loudness-per-frequency-band feature at 0% through the sample, 5%,
   and 50%.)

 - Drums, while being very percussive instruments, [can still be
   **tuned**](https://en.wikipedia.org/wiki/Drum_tuning) to various pitches. To
   quantify this tuning and help our algorithm use it as input, we can take the
   [fundamental frequency](https://en.wikipedia.org/wiki/Fundamental_frequency)
   of the sample to help the algorithm distinguish between high drums and low
   drums.

These are just some of the many features that might be useful for solving our
classification problem, but let's start with these four and see how far we get.

As with all machine learning problems, to teach the machine to do something,
you have to have some sort of _training data_. In this case, I'm going to use
a handful of samples - roughly 20-30 from each instrument - from the tens of
thousands of samples I have in my sample collection. When choosing these
samples, I want to find:

 - samples that are representative of the different types of each instrument
   (e.g.: a few acoustic kick drums, some electronic kick drums, some
   beatboxed kicks, and so on)
 - samples from different sources that might have different biases that
   humans have a harder time picking up on (e.g.: are all samples from one
   sample pack the exact same length? what about the same fundamental
   frequency?)
 - samples of things that _aren't_ drums, so that the algorithm can learn
   when a sample falls into the "something else" bucket

I put together a list of these samples - 108 files, roughly 50 megabytes of
sample data, in five separate folders: `kick`, `snare`, `hat`, `percussion`,
and `other`. (Most of these samples are from FreeSound.com and are licensed
under a Creative Commons Attribution License, so special thanks to
[waveplay](https://freesound.org/people/waveplay/),
[Seidhepriest](https://freesound.org/people/Seidhepriest/),
and [quartertone](https://freesound.org/people/quartertone) for making their
samples available for free!)

Now that we've got some data to train on, let's write some code to perform the
feature extraction mentioned earlier. These features aren't super hard for us
to calculate, but they're also not super simple, so I've written some code
below to extract them by using
[`librosa`](https://librosa.github.io/librosa/), a wonderful Python library
for audio analysis by the wonderful [Brian McFee](http://bmcfee.github.io/)
et al.

(Editor's note: in between this paragraph and the next block of code,
literally three hours passed as the author wrangled with Python, pip,
and various dependency issues.)

<Insert feature_extract.py snippets here>

Alright, so now we've got a number of features extracted from each sample,
all being saved as one large JSON file. We can think of these features
as measurements we're taking of the samples, without having to use the
entire contents of the samples themselves. (And that's very true in this
case - we started with over 50 megabytes of samples, but the features
themselves are only 150 kilobytes - that's more than 300 times smaller.)

Now, we can take these features and give them to a machine learning
algorithm and have it learn from them. But hold on a sec - let's get
specific about which algorithm we're talking about, and about what
learning means in this context.

We're going to use an algorithm called a [decision
tree](https://en.wikipedia.org/wiki/Decision_tree) in this post, which is a
commonly used machine learning algorithm that _doesn't_ involve some of the
buzzwords that you may have heard, like "neural networks," "deep learning," or
"artificial intelligence." A decision tree is a **system that splits data into
categories by learning thresholds for each feature** in a recursive way. (If
that's confusing, don't worry too much about it - but checkout [R2D3's amazing
visual example of how decision trees work](http://www.r2d3.us/visual-intro-to-
machine-learning-part-1/) if you're curious).

< insert classifier.py >

In this case, `classifier.py` _trains a model_ by creating a decision tree -
which _is_ our model - whose weights are statistically determined by the data
that we pass in. Again, the specifics aren't necessary to understand for the
rest of this post, but here's what that model looks like when visualized:

< insert model diagram >

Now, if we run `classifier.py`, we should see two lists: one of the **training**
accuracy (how well the model predicted the kind of sample for samples that it
saw during training) and the **test** accuracy (now well the model predicted
samples that it hadn't seen before). As we might expect, the training accuracy
is 100% - that data was used to create the model! But of the samples that the
model hadn't seen before, it only got three out of four guesses (75%) correct.

This is an example of what's called _overfitting_ - our model has trained
itself to be overly specific and be completely accurate for data that it's
seen before, but it has trouble when it sees data that's new to it. In some
sense, this is similar to how humans learn; when someone sees something new
that they hadn't seen in school or heard about before, they're bound to make
mistakes.

To avoid overfitting our model, we could take a number of approaches:

 - We could tune the algorithm's parameters to try to force it to be less
   specific. This is a good place to start, especially with decision tree
   algorithms.
 - We could change our feature calculation to give more data to the
   algorithm, possibly introducing data that seems unintuitive to humans
   but would mathematically help solve our classification problem.
 - We could add more (and more varied) data so that the decision tree
   algorithm can create a more general tree, assuming that the existing set
   of data isn't complete enough.

All three of these are valid approaches, and they're also left up to the
reader to investigate. We could also try other classification methods instead
of using a decision tree, although surprisingly a naïve decision tree works
pretty well for this problem.

So! We've built a machine learning classifier for drum samples. _That's kinda
cool._ There are a couple things to note about this system:

 - We do our machine learning training on _features_, rather than the
   audio data itself. This means that if we wanted to write a program to
   classify new, unknown samples against this model, it would first have
   to run the sample through the same logic that's in `feature_extract.py`
   before it would be compatible with the model.
 - The current model is held _in memory_ and never written out to disk. This
   is somewhat impractical, and in a real-world machine learning system,
   you'd likely save the model as a separate file that you could then pass
   around and use in different situations. (In many popular machine learning
   systems, models are trained routinely on up to _terabytes_ of input data,
   rather than the 40 megabytes we used here, so storing the outputted model
   on disk is very necessary.)
 - We're currently training this model on around 150 samples, which gives
   okay results and allows us to test this model training in seconds rather
   than minutes or hours. We could try training this on _literally all of the
   samples available to us_, which maight give much better results. (In tests
   on my entire sample library, I was able to get up to 90% accuracy, which
   is pretty good for a simple decision tree.)
 - This model is a _classifier_, which means that while it can put samples
   into buckets of sorts (and even give probability of a sample being in a
   bucket) it can't tell you how much, say, a snare sounds like a kick. If
   you want to place your sounds along a continuous scale rather than into
   buckets, you'll need another kind of machine learning algorithm.
 - The algorithm used by `scikit` uses a random variable to choose how to
   create its decision tree. If this model was to be used in production,
   this random number generator should be
   [seeded](https://en.wikipedia.org/wiki/Random_seed)
   to allow for exactly reproducible results, which makes it easier to test,
   debug, and use the model.

If you've got your own sample library, or want to give this problem a try with
samples you've found online, go for it! All of the code from this blog post is
available
[here on Github](https://github.com/psobot/machine-learning-for-drummers),
and you can pop in your own sample packs and have fun. Some other things to try:

 - Try using different features. `librosa` is very advanced and exposes many
   parameters about the audio it's analyzing - choose as many features as you'd
   like and try to improve your accuracy!
 - Try tuning the algorithm used for machine learning. Scikit's
   `DecisionTreeClassifier` has a lot of options that might improve accuracy by
   a lot. (If you end up trying to optimize this automatically, that's called
   [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
   and is its own field of study within machine learning.)
 - Try throwing new kinds of audio files at this system to see what breaks.
   My training and test datasets didn't include any longer audio files,
   full songs, podcasts, or other audio files that you might find. See how
   those files work with this model and see if you can improve it to handle
   those cases better.
