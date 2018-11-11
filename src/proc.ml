
let tagger_pid = ref 0;;

let recvall sock =
    let len = input_binary_int sock in
    let buf = Bytes.create len in
    really_input sock buf 0 len;
    buf

let communicate input filename =
    (* if !tagger_pid = 0 then                                      *)
    (*     tagger_pid := Unix.create_process "python"               *)
    (*             [|"python"; "server.py"; "models/tri_headfirst"; *)
    (*               "--filename"; "/tmp/test2"|]                   *)
    (*             Unix.stdin Unix.stdout Unix.stderr;              *)
    (* Unix.sleep 5; *)
    let sock = Unix.socket Unix.PF_UNIX Unix.SOCK_STREAM 0 in
    let addr = Unix.ADDR_UNIX filename in
    Unix.connect sock addr;
    let sock_out = Unix.out_channel_of_descr sock in
    let sock_in = Unix.in_channel_of_descr sock in
    let input_length = Bytes.length input in
    output_binary_int sock_out input_length;
    output_bytes sock_out input;
    flush sock_out;
    recvall sock_in


(*
let () = 
    let input = "this is an example test .|Hi ." in
    p input "/tmp/test4";
    ()
    (* Unix.kill !tagger_pid Sys.sigterm; () *)
*)
