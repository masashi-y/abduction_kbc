(** message.proto Types *)



(** {2 Types} *)

type predicate = {
  str : string;
  nargs : int;
}

type candidate = {
  pred1 : predicate option;
  pred2 : predicate option;
  rel : string;
  score : float;
}

type rank = {
  list : candidate list;
}

type echo = {
  msg : string;
  rank : rank option;
}


(** {2 Default values} *)

val default_predicate : 
  ?str:string ->
  ?nargs:int ->
  unit ->
  predicate
(** [default_predicate ()] is the default value for type [predicate] *)

val default_candidate : 
  ?pred1:predicate option ->
  ?pred2:predicate option ->
  ?rel:string ->
  ?score:float ->
  unit ->
  candidate
(** [default_candidate ()] is the default value for type [candidate] *)

val default_rank : 
  ?list:candidate list ->
  unit ->
  rank
(** [default_rank ()] is the default value for type [rank] *)

val default_echo : 
  ?msg:string ->
  ?rank:rank option ->
  unit ->
  echo
(** [default_echo ()] is the default value for type [echo] *)
