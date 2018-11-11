[@@@ocaml.warning "-27-30-39"]

type predicate_mutable = {
  mutable str : string;
  mutable nargs : int;
}

let default_predicate_mutable () : predicate_mutable = {
  str = "";
  nargs = 0;
}

type candidate_mutable = {
  mutable pred1 : Message_types.predicate option;
  mutable pred2 : Message_types.predicate option;
  mutable rel : string;
  mutable score : float;
}

let default_candidate_mutable () : candidate_mutable = {
  pred1 = None;
  pred2 = None;
  rel = "";
  score = 0.;
}

type rank_mutable = {
  mutable list : Message_types.candidate list;
}

let default_rank_mutable () : rank_mutable = {
  list = [];
}

type echo_mutable = {
  mutable msg : string;
  mutable rank : Message_types.rank option;
}

let default_echo_mutable () : echo_mutable = {
  msg = "";
  rank = None;
}


let rec decode_predicate d =
  let v = default_predicate_mutable () in
  let continue__= ref true in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
    ); continue__ := false
    | Some (1, Pbrt.Bytes) -> begin
      v.str <- Pbrt.Decoder.string d;
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(predicate), field(1)" pk
    | Some (2, Pbrt.Varint) -> begin
      v.nargs <- Pbrt.Decoder.int_as_varint d;
    end
    | Some (2, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(predicate), field(2)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  ({
    Message_types.str = v.str;
    Message_types.nargs = v.nargs;
  } : Message_types.predicate)

let rec decode_candidate d =
  let v = default_candidate_mutable () in
  let continue__= ref true in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
    ); continue__ := false
    | Some (1, Pbrt.Bytes) -> begin
      v.pred1 <- Some (decode_predicate (Pbrt.Decoder.nested d));
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(candidate), field(1)" pk
    | Some (2, Pbrt.Bytes) -> begin
      v.pred2 <- Some (decode_predicate (Pbrt.Decoder.nested d));
    end
    | Some (2, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(candidate), field(2)" pk
    | Some (3, Pbrt.Bytes) -> begin
      v.rel <- Pbrt.Decoder.string d;
    end
    | Some (3, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(candidate), field(3)" pk
    | Some (4, Pbrt.Bits64) -> begin
      v.score <- Pbrt.Decoder.float_as_bits64 d;
    end
    | Some (4, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(candidate), field(4)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  ({
    Message_types.pred1 = v.pred1;
    Message_types.pred2 = v.pred2;
    Message_types.rel = v.rel;
    Message_types.score = v.score;
  } : Message_types.candidate)

let rec decode_rank d =
  let v = default_rank_mutable () in
  let continue__= ref true in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
      v.list <- List.rev v.list;
    ); continue__ := false
    | Some (1, Pbrt.Bytes) -> begin
      v.list <- (decode_candidate (Pbrt.Decoder.nested d)) :: v.list;
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(rank), field(1)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  ({
    Message_types.list = v.list;
  } : Message_types.rank)

let rec decode_echo d =
  let v = default_echo_mutable () in
  let continue__= ref true in
  while !continue__ do
    match Pbrt.Decoder.key d with
    | None -> (
    ); continue__ := false
    | Some (1, Pbrt.Bytes) -> begin
      v.msg <- Pbrt.Decoder.string d;
    end
    | Some (1, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(echo), field(1)" pk
    | Some (2, Pbrt.Bytes) -> begin
      v.rank <- Some (decode_rank (Pbrt.Decoder.nested d));
    end
    | Some (2, pk) -> 
      Pbrt.Decoder.unexpected_payload "Message(echo), field(2)" pk
    | Some (_, payload_kind) -> Pbrt.Decoder.skip d payload_kind
  done;
  ({
    Message_types.msg = v.msg;
    Message_types.rank = v.rank;
  } : Message_types.echo)

let rec encode_predicate (v:Message_types.predicate) encoder = 
  Pbrt.Encoder.key (1, Pbrt.Bytes) encoder; 
  Pbrt.Encoder.string v.Message_types.str encoder;
  Pbrt.Encoder.key (2, Pbrt.Varint) encoder; 
  Pbrt.Encoder.int_as_varint v.Message_types.nargs encoder;
  ()

let rec encode_candidate (v:Message_types.candidate) encoder = 
  begin match v.Message_types.pred1 with
  | Some x -> 
    Pbrt.Encoder.key (1, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_predicate x) encoder;
  | None -> ();
  end;
  begin match v.Message_types.pred2 with
  | Some x -> 
    Pbrt.Encoder.key (2, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_predicate x) encoder;
  | None -> ();
  end;
  Pbrt.Encoder.key (3, Pbrt.Bytes) encoder; 
  Pbrt.Encoder.string v.Message_types.rel encoder;
  Pbrt.Encoder.key (4, Pbrt.Bits64) encoder; 
  Pbrt.Encoder.float_as_bits64 v.Message_types.score encoder;
  ()

let rec encode_rank (v:Message_types.rank) encoder = 
  List.iter (fun x -> 
    Pbrt.Encoder.key (1, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_candidate x) encoder;
  ) v.Message_types.list;
  ()

let rec encode_echo (v:Message_types.echo) encoder = 
  Pbrt.Encoder.key (1, Pbrt.Bytes) encoder; 
  Pbrt.Encoder.string v.Message_types.msg encoder;
  begin match v.Message_types.rank with
  | Some x -> 
    Pbrt.Encoder.key (2, Pbrt.Bytes) encoder; 
    Pbrt.Encoder.nested (encode_rank x) encoder;
  | None -> ();
  end;
  ()
