[@@@ocaml.warning "-27-30-39"]


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

let rec default_predicate 
  ?str:((str:string) = "")
  ?nargs:((nargs:int) = 0)
  () : predicate  = {
  str;
  nargs;
}

let rec default_candidate 
  ?pred1:((pred1:predicate option) = None)
  ?pred2:((pred2:predicate option) = None)
  ?rel:((rel:string) = "")
  ?score:((score:float) = 0.)
  () : candidate  = {
  pred1;
  pred2;
  rel;
  score;
}

let rec default_rank 
  ?list:((list:candidate list) = [])
  () : rank  = {
  list;
}

let rec default_echo 
  ?msg:((msg:string) = "")
  ?rank:((rank:rank option) = None)
  () : echo  = {
  msg;
  rank;
}
