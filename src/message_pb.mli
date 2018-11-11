(** message.proto Binary Encoding *)


(** {2 Protobuf Encoding} *)

val encode_predicate : Message_types.predicate -> Pbrt.Encoder.t -> unit
(** [encode_predicate v encoder] encodes [v] with the given [encoder] *)

val encode_candidate : Message_types.candidate -> Pbrt.Encoder.t -> unit
(** [encode_candidate v encoder] encodes [v] with the given [encoder] *)

val encode_rank : Message_types.rank -> Pbrt.Encoder.t -> unit
(** [encode_rank v encoder] encodes [v] with the given [encoder] *)

val encode_echo : Message_types.echo -> Pbrt.Encoder.t -> unit
(** [encode_echo v encoder] encodes [v] with the given [encoder] *)


(** {2 Protobuf Decoding} *)

val decode_predicate : Pbrt.Decoder.t -> Message_types.predicate
(** [decode_predicate decoder] decodes a [predicate] value from [decoder] *)

val decode_candidate : Pbrt.Decoder.t -> Message_types.candidate
(** [decode_candidate decoder] decodes a [candidate] value from [decoder] *)

val decode_rank : Pbrt.Decoder.t -> Message_types.rank
(** [decode_rank decoder] decodes a [rank] value from [decoder] *)

val decode_echo : Pbrt.Decoder.t -> Message_types.echo
(** [decode_echo decoder] decodes a [echo] value from [decoder] *)
