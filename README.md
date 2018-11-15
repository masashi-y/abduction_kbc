# Codebase of _Combining Axiom Injection and Knowledge Base Completion for Efficient_, AAAI 2019

[paper](/tmp)

Requirements
* Coq 8.6 (Sadly, the program must be compiled from the source! No opam, apt or homebrew ...)
* Python 3

Note that [ccg2lambda](https://github.com/mynlp/ccg2lambda) uses Coq 8.4 but our code only compiles with 8.6.

I have checked this program runs properly under these environments:
Ubuntu and macOS Sierra with
* ocaml: 4.05
* camlp4: 4.05
* camlp5: 7.03
* ocamlfind: 1.7.3

### Build
```sh
$ coq_makefile -f _CoqProject -o Makefile
$ make
$ coqc coqlib.v
$ pip install protobuf
```

Then add these lines to $HOME/.coqrc:

```
Add Rec LoadPath "/path/to/abductionKB/theories/" as Abduction.
Add ML Path "/path/to/abductionKB/src".
```

### Check if the build is successful

```sh
# Download a pretrained model on WordNet.
$ wget http://cl.naist.jp/~masashi-y/resources/abduction/model_wordnet.config
$ cd src
# Use --daemon or run the program in other terminal
$ python server.py run --threshold 0.4 --daemon ../model_wordnet.config 
$ cd .. 
# test.v: a theorem of "a man and a woman hike through a wooded area" => "a man and a woman walk through a wooded area"
$ cat test.v | coqtop
# Now it successfully solves the problem by injecting lexical axioms!
```

### Usage of `server.py`

```
$ python server.py
usage: server.py run [-h] [--filename FILENAME] [--daemon]
                     [--threshold THRESHOLD]
                     model
```

* `filename`: UNIX domain socket address used for the communication between Python and Coq. Defaults to `/tmp/py_server`.
* `daemon`:  if set, the server is daemonized.
* `threshold`: Only triplets whose scores ared larger than this value are adopted as axioms. We recommend setting this to `0.4`. 
* `model`: path to model config file

### Usage of abduction tactic

```coq
(* This must be done firstly *)
Require Export Abduction.Abduction.

(* Make connection to a Python server. *)
(* NeCheck can be set to disable if the connection is OK  *)
Abduction Set Server [NoCheck] server_name.

(* Show cached axioms generated so far. *)
Abduction Show Cache.

(* Debugging mode with higher verbosity *)
Abduction (Set|Reset) Debug.

(* Show the connected server address *)
Abduction Get Server.

Parameter _hike : Event -> Prop.
Parameter _walk : Event -> Prop.
Goal forall x, _hike x -> _walk x.

intros.
(* 1 subgoal
 *   x : Event
 *   H : _hike x
 *   ============================
 *   _walk x    *)

abduction.
(* 1 subgoal
 *  x : Event
 *  H : _hike x
 *  NL_axiom1 : impl_fun1 Event _hike _walk
 *  ============================
 *  _walk x     *)  
```

#### Pretrained models

- [model](http://cl.naist.jp/~masashi-y/resources/abduction/model_wordnet.config) trained on WordNet
- [model](http://cl.naist.jp/~masashi-y/resources/abduction/model_wordnet_verbocean.config) trained on WordNet and VerbOcean

### Interactively work with ComplEx model

```sh
$ python predict.py run --model model.config
>> star, galaxy
  1: star.n	galaxy.n	member_holonyms	4.305
  2: star.n	galaxy.n	antonyms	0.068
  3: star.n	galaxy.n	attributes	-0.716
  4: star.n	galaxy.n	member_meronyms	-2.288
  ...
```

### TODO?
* https://sympa.inria.fr/sympa/arc/coq-club/2017-11/msg00029.html

### Cite

to do

### Acknowledgement

- [mana-ysh/knowledge-graph-embeddings](https://github.com/mana-ysh/knowledge-graph-embeddings)
- [TimDettmers/ConvE](https://github.com/TimDettmers/ConvE)
- [easonnie/ResEncoder](https://github.com/easonnie/ResEncoder)

### Contact