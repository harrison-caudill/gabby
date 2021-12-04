Gabby
=====

This package started with a simple request from Brian Weeden at the
Secure World Foundation: animated Gabbard plot of the debris of the
Fengyun 1C incident from 2007.  I may have gotten a little carried
away, especially since the data are going to be useful in the OSA's
farewell publication.


Things that Gabby will do
=========================

Gabby was originally written for the single purpose of animated
gabbard plots (currently posted to the OSA website).  However, I have
since adapted the framework for other purposes.  Several others are
somewhat trivial, but do not have command-line utilities implemented
(e.g. producing a static PNG file of the number of tracked pieces of
debris between two date ranges resulting from a specific event).


Animated Gabbard Plots
----------------------

Using imported TLE data, gabbard plots are generated at regular time
intervals and exported as .png files.  `ffmpeg` is then used to encode
those pngs into an mp4.

[Example output](https://www.youtube.com/playlist?list=PLcuXVs-1Tw_8kIBI3pwtJMicPGbkqHwdQ)


Upcoming: Forward Propagation
-----------------------------

Using first derivative statistics for APT as a function of the apogee
and perigee, propagate debris forward in time and get an idea of what
the next N years may look like.  This feature is intended for things
like examining the potential future consequences of the Nudol attack
or discussing the effects of continued use of ASAT weapons.



Workflow
========

Download the raw TLE data from Space-Track.org
----------------------------------------------

This will be in large zip files whose TLEs will include the proper
international designators.  If you look in `cfg/asat.cfg` you'll find
a series of shasums that can be used to ensure you're working with the
same data that I am.  I don't currently use the shasums in the code,
but if you bug me, I could probably add that.  Or you could...just
sayin'...


Snarf in the Data
-----------------

Gabby really only pays attention to 3 numbers for any given object at
any given moment: APT.  Apogee (A), Perigee (P), and Orbital Period
(T).  The ingestion process will parse the TLE files, compute the APT
values, and stash them in a local lmdb (light memory-mapped DB).  lmdb
is pretty similar to the old Berkeley DB, but with a friendlier
license.  See the note below about oblate sphereoids below.


Plot the Frames
---------------

Gabby will use the DB to produce the png files.  Matplotlib isn't
particularly threadsafe, so it uses the multiprocessing module in
python to launch separate processes.  It takes about 100-200ms per
image.


Build the video
---------------

Pretty trivial `ffmpeg` command, but there's a convenience script:
`bin/producer`



Oblate Sphereoids are Annoying
==============================

I have maybe 3 friends that can truly understand this little rant, and
the internet has been unable to assist so far.  If you have insight
into this little problem, please reach out.

The Earth is not a sphere.  It isn't even close.  Even if a giant
buffing wheel were to polish all the mountains down, and fill up all
the valleys, it still wouldn't be a sphere.  As Newton predicted, the
spinning of the Earth on its axis causes it to bulge at the equator
like a speed governer on an old timey steam engine.  That creates two
problems that are worthy of consideration.

 1. If you have a satellite that is a perfect polar orbit and a
    perfect circle (relative to the center of the Earth), then as it
    approaches the equator (where the planet bulges), the AGL (Above
    Ground Level) orbital altitude will decrease.  So the meaning of
    perigee and apogee can get a little fuzzy.  If it is computed by
    subtracting the mean Earth radius from the minimum radius from
    Earth's center, then you may get the wrong answer.

 2. The oblateness of the planet causes a periodic change in the
    gravitational potential energy (I think).  If you use a classical
    Keplerian approximation for computing the Apogee and Perigee of a
    satellite from its TLE using only the eccentricity and period,
    then you end up with a variation (can be several km) in the
    computed APT values dependening upon the phase of the orbit at
    which the observation was made.

Code Structure
==============

Gabby was not written particularly well, as it was originally designed
to be a one-off.  I may or may not clean up code as I go, write unit
tests, docstrings, etc.  One note about the code structure: it *is*
designed to be performant.  When originally written, it would have
taken something like 24 hours to import all the TLE data through 2021
from Space-Track.org -- it takes 30 minutes on my laptop at the time
of writing this.  It was also taking 2 hours to produce a single
animation (~2 seconds per image with memory leaks preventing high
resolution) whereas it now takes about 100-200ms/frame.  Some of the
decisions made to improve performance make it a bit less readable.

The raw data to plot is first pre-computed in numpy arrays so that we
can make use of the inate parallelism there.  That design choice,
however, means that you need enough RAM to hold N different copies of
all the data to plot where N is the number of separate threads.
Realistically, this approach works pretty well (at least on my
laptop).

NOTE: There's a hard-coded matplotlib command in there that changes
      the rendering engine to `TkAgg` to prevent memory leaks.  It
      also disables interactive mode.  This step is only thought to be
      necessary when you're working with macos.


Local DB
--------

A local database (lmdb) is kept which matches the time/fragment to the
APT values.

 * `DB_NAME_IDX`: `<des>,<ts>: fff`
 * `DB_NAME_GABBY`: `<ts>,<des>: fff`
 * `DB_NAME_SCOPE`: `<des>: ii`

 * `DB_NAME_TLE`: Unused at the moment, but the idea is to have the
                  ability to record all other structured data from the
                  TLE.

The index and gabby databases have duplicated data, just with the
values in the keys reversed.  If you want to walk through the entire
history of space travel, then use the gabby table.  If you want to map
only specific fragements, use the index table.  The APT values are
encoded as floats using the python `struct` module.

The scope table lets you know when a piece comes into scope, and when
it leaves scope.  This one is useful for things like determining how
many pieces of debris from a given event are in orbit at any given
moment.

The timestamps are classical unix integer timestamps from the epoch in
UTC.  The designators are all international designator, so
<2-digit-year><3-digit-launch-number><3-character-piece>.


Files
-----

 * `bin/plot`: Produces the png files for the animated gabby plots

 * `bin/producer`: Fires up `ffmpeg` to encode the png files into a
   single video.

 * `gabby/*`: Just what you'd think

 * `cfg/asat.cfg`: The gabby config file I used to generate the videos
   on Youtube.


To Do
=====

 * Add an option during the snarf phase to, for a subset of the
   TLEs, use an SGP4 propagator to simulate one complete orbit and
   observe the min and max values for a better approximation of the
   apogee and perigee.

 * Ideally, find an analytical approximateion of the form APT(TLE)
   that takes into account J2 and can be run quickly enough to
   properly import APT values for all TLEs

 * Compute the frequency histogram of A'(A,P), P'(A,P), and T'(A,P)

 * Expand the PMF with resample_poly (or resample, who cares if we
   do a polyphase decomp on this one).

 * Reverse-propagate all samples for a given breakup event from
   initial observation back to a hypothesized start at the time of
   the breakup.

 * Forward-propagate all observations for the recent nudol test
   using the frequency distributions observed from past breakup
   events.

 * Look into normalizing the data for the B* term in the TLE.

 * Import prior data for solar activity and normalize the frequency
   distributions for past solar activity.

 * Incorporate expected future solar activity into the forward
   propagation.

 * Clean up the data.  For example designator 93036ABC has about a
   13.5 mean motion except for one single observation which is
   closer to 11.25.  That kind of data quality causes some pretty
   interesting behavior in the generated videos.

 * Add historical solar activity into the lower plot on the gabby
   plot indicating level of solar activity, probably as background
   shading or something.
