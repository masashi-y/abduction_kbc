open Names
open Evd
open Coqlib
open Message_types
open Message_pb
open Utils

DECLARE PLUGIN "abduction"

let contrib_name = "abductionKB"

let pp_constr fmt x = Pp.pp_with fmt (Printer.pr_constr x)
let pp_type fmt x = Pp.pp_with fmt (Printer.pr_ltype x)
let pp_constant fmt x = Pp.pp_with fmt (Printer.pr_constant (Global.env ()) x)
let pp_sort fmt (ev_map, x) = Pp.pp_with fmt (Printer.pr_sort ev_map x)
(* let pp_econstr fmt x = Pp.pp_with fmt (Printer.pr_econstr x) *)
(* let pp_constrexpr fmt x = Pp.pp_with fmt (Ppconstrsig.Pp.pr_constr_expr x) *)

let show_constr x = Pp.string_of_ppcmds (Printer.pr_constr x)
let show_type x = Pp.string_of_ppcmds (Printer.pr_ltype x)
let show_constant x = Pp.string_of_ppcmds (Printer.pr_constant (Global.env ()) x)
let show_sort (ev_map, x) = Pp.string_of_ppcmds (Printer.pr_sort ev_map x)
let show_types xs = "[" ^ (String.concat ", " @@ List.map show_type xs) ^ "]"
(* let show_econstr x = Pp.string_of_ppcmds (Printer.pr_econstr x) *)

let find_constant contrib dir s =
    Universes.constr_of_global @@ Coqlib.find_reference contrib dir s

let init_constant dir s = find_constant contrib_name dir s

let root_path = ["coqlib"]
let entity_type = lazy (init_constant root_path "Entity")
let event_type = lazy (init_constant root_path "Event")
let subj = lazy (init_constant root_path "Subj")

let path = ["Abduction"; "Abduction"]

let imp = lazy (init_constant root_path "imp1")
let devil = lazy (init_constant path "proof_admitted")

let impl1 = lazy (init_constant path "impl_fun1")
let impl2 = lazy (init_constant path "impl_fun2")
let back1 = lazy (init_constant path "backoff_fun1")
let back2 = lazy (init_constant path "backoff_fun2")
let ant_impl1 = lazy (init_constant path "ant_impl_fun1")
let ant_impl2 = lazy (init_constant path "ant_impl_fun2")
let ant_impl1_subj = lazy (init_constant path "ant_impl_fun1_subj")


let new_axiom_name =
    let ptr = ref 0 in
    fun () -> ptr := !ptr + 1;
             Id.of_string (Format.sprintf "NL_axiom%i" !ptr)


module Predicate =
struct
    type t = Term.constr * Term.types list * int
    
    let show (c, t, n) = Printf.sprintf "(%s: (%s), nargs: %i)" (show_constr c) (show_type t) n
    let compare v w = compare v w
    let is_ccg2lambda_term (_, p, _) = p.[0] = '_'
end


module H =
struct
    include Hashtbl

    let find_through ds k =
        let f res d = match res with
            | Some _ -> res
            | None   ->
                try Some (find d k)
                with Not_found -> None
        in match List.fold_left f None ds with
        | Some v -> v
        | None -> raise Not_found
end

(* global variables *)
let server_address = ref ""
and debug = ref false
and cache = H.create 100

let construct_message preds concls =
    let rank, cached = H.fold (fun cstr (_, _, nargs) tup ->
        let cp = default_predicate ~str:cstr ~nargs () in
        H.fold (fun pstr (_, _, nargs) (rank, cached) -> 
            try
                (rank, (H.find cache (cstr, pstr)) @ cached)
            with Not_found ->
                let pp = default_predicate ~str:pstr ~nargs () in
                let c = default_candidate ~pred1:(Some cp) ~pred2:(Some pp) () in
                H.add cache (cstr, pstr) [];
                (c :: rank, cached)
        ) preds tup
    ) concls ([], [])
   in (default_rank ~list:rank (), cached)

let send_message msg file =
    let encoder = Pbrt.Encoder.create () in
    encode_echo (default_echo ~rank:(Some msg) ()) encoder;
    let res = Proc.communicate (Pbrt.Encoder.to_bytes encoder) file in
    let dec = decode_echo (Pbrt.Decoder.of_bytes res) in
    match dec.rank with
    | Some dec -> dec
    | None -> failwith "error in send_message"

(* if c : Event -> Entity -> Prop, then output [Event; Entity] *)
let get_arg_types ev_map c (types : Term.types list) =
    let env = (Global.env ()) in
    let c_type = Typing.unsafe_type_of env ev_map c in
    let fail () = failwith (!%"unexpected input: %s\n" (show_type c_type)) in
    let rec f c types =
        match Term.kind_of_term c with
        | Term.Prod (_, a, b) -> f b (a :: types)
        | Term.Const _ | Term.Sort _ -> c :: types
        | _ -> fail ()
    in List.rev (match f c_type [] with
            _ :: ls -> ls | _ -> fail ())


let rec decomp_constr c ev_map terms =
    match Term.kind_of_term c with
    (* | Term.Const c -> terms *)
        (* let pred = show_constant (c, ev_map) in *)
        (* SS.add (Term.Const c, pred, 0) terms *)
    (* | Term.Cast (c,_,t) *)
    | Term.Prod (Name x,t,c)
    | Term.Lambda (Name x,t,c) ->
            (* print_endline (Id.to_string x); *)
            decomp_constr t ev_map terms |> decomp_constr c ev_map
    | Term.LetIn (Name x,b,t,c) ->
            (* print_endline (Id.to_string x); *)
            decomp_constr b ev_map terms |> decomp_constr t ev_map |> decomp_constr c ev_map
    | Term.App (c,l) as term ->
        let pred = show_type c in
        (* assume all ccg2lambda predicates starts with "_" *)
        if pred.[0] = '_' then begin
            let nargs = Array.length l in
            let arg_types = (get_arg_types ev_map c []) in
            H.add terms pred (c, arg_types, nargs)
        end;
        Array.fold_right (fun a s -> decomp_constr a ev_map s) l terms

    (* | Term.Proj (p,c) -> decomp_constr p ev_map *)
    (* | Term.Evar (_,l) -> Array.fold_left f acc l *)
    (* | Term.Case (_,c,bl) -> Array.fold_left f (f (f acc p) c) bl *)
    (* | Term.Fix (_,(lna,tl,bl)) -> *)
    (*       Array.fold_left2 (fun acc t b -> f (f acc t) b) acc tl bl *)
    (* | Term.CoFix (_,(lna,tl,bl)) -> *)
    (*       Array.fold_left2 (fun acc t b -> f (f acc t) b) acc tl bl *)
    (* | Term.Const (c, _) ->                               *)
    (*     print_endline (show_constant (c, ev_map)); terms *)
    (* | Term.Rel i -> print_int i; terms *)
    (* | Term.Var i -> print_endline (Id.to_string i); terms *)
    | _ -> terms
    (* | Term.(Rel _ | Meta _ | Var _ | Sort _ | Ind _ | Construct _) -> () *)


let have_same_arg_types l1 l2 =
    List.length l1 = List.length l2 &&
        List.fold_left2 (fun b t1 t2 -> b && (Constr.equal t1 t2)) true l1 l2
(*
let do_assumptions_bound_univs env id ty =
    let pl = None in
    (* let env = Global.env () in *)
    let ctx = Evd.make_evar_universe_context env pl in
    let evdref = ref (Evd.from_ctx ctx) in
    (* let ty, impls = interp_type_evars_impls env evdref c in *)
    let nf, subst = Evarutil.e_nf_evars_and_universes evdref in
    let ty = nf ty in
    let vars = Universes.universes_of_constr ty in
    let evd = Evd.restrict_universe_context !evdref vars in
    let pl, uctx = Evd.universe_context evd in
    (* let pl, uctx = Evd.universe_context ?names:pl evd in *)
    let uctx = Univ.ContextSet.of_context uctx in
    let kind = (Decl_kinds.Global, false, Decl_kinds.Logical) (* Axiom *) in
    let nl = Vernacexpr.NoInline in
    let (_, _, st) = Command.declare_assumption false kind (ty, uctx) pl [] false nl id in
    st
*)

let ij_hello = Proofview.Goal.nf_enter { enter = begin fun gl ->
    let concl = Tacmach.New.pf_concl gl
    and sigma = ref (Tacmach.New.project gl)
    and env = Tacmach.New.pf_env gl in
    (* let _ = Format.printf "print test:\n%a\n" show_type concl in *)

    (* collect terms from goal and context *)
    let open Context.Named in
    let f decl terms = match decl with
        | Declaration.LocalAssum (_, constr)
        | Declaration.LocalDef (_, constr, _) ->
                decomp_constr constr !sigma terms in
    let concl_preds = decomp_constr concl !sigma (H.create 100) in
    let premise_preds = fold_outside f (Proofview.Goal.hyps gl) ~init:(H.create 100) in
    let rank, cached = construct_message premise_preds concl_preds in
    let event_type = Lazy.force event_type in
    let entity_type = Lazy.force entity_type in
    let axioms = match rank.list with
        | [] -> cached
        | _ -> begin
        let res = send_message rank !server_address in
        let f = H.find_through [premise_preds; concl_preds] in
        let axioms = List.fold_left (fun init -> function
        | { pred1=Some pred1; pred2=Some pred2; rel } ->
            let s1, s2 = (pred1.str, pred2.str) in
            let c1, ts1, nargs = f s1
            and c2, ts2, _____ = f s2 in
            let have_same_args = have_same_arg_types ts1 ts2 in
            let g, args = match rel, nargs with
                (* | "anto", 1 when Constr.equal (List.hd ts1) event_type -> ant_impl1_subj, [entity_type; c2; c1; Lazy.force subj] *)
                (* | "anto", 1 when Constr.equal (List.hd ts1) entity_type -> ant_impl1, [c2; c1] *)
                | "anto", 1   -> ant_impl1, [c2; c1]
                | "anto", 2 -> ant_impl2, [c2; c1]
                | "entail", 1 -> impl1, [c2; c1]
                | "entail", 2 -> impl2, [c2; c1]
                | "back", 1 -> back1, [c2; c1]
                | "back", 2 -> back2, [c2; c1]
                | "hypo", 1 -> impl1, [c1; c2]
                | "hypo", 2 -> impl2, [c1; c2]
                | _ -> failwith
        (!%"not supported abduction pattern: (%s, %i)\n" rel nargs) in
            if !debug then begin
                Format.eprintf
                "%s: %s\t%s: %s\trel: %s\thave same args: %s\n"
                    s1 (show_types ts1) s2 (show_types ts2)
                    rel (if have_same_args then "YES" else "NO")
            end;
            if not have_same_args then init
            else begin
                let args = Array.of_list (ts1 @ args) in
                let term = Term.mkApp (Lazy.force g, args) in
                sigma := fst (Typing.type_of ~refresh:true env !sigma term);
                let ls = match H.find_opt cache (s1, s2) with
                | Some ls -> term :: ls
                | None -> [term] in
                H.replace cache (s1, s2) ls;
                term :: init
            end
        | _ -> failwith "fail in ij_hello"
        ) [] res.list in axioms @ cached
        end in
        (*
        | { pred1=Some pred1; pred2=Some pred2; rel="anto" } ->
              (Lazy.force ant_impl1, [| Lazy.force entity_type; f pred2; f pred1 |])
        | { pred1=Some pred1; pred2=Some pred2; rel="hypo" } ->
              (Lazy.force impl1, [| Lazy.force entity_type; f pred2; f pred1 |])
        *)

    if !debug then begin
        Format.printf "num response %i\n" (List.length axioms);
        (* Tacmach.New.of_old (fun gl ->
            Tacticals.tclIDTAC_MESSAGE (Pp.str "Hello, world!\n") gl) (Proofview.Goal.assume gl) *)
    end;
    (*
    let preds = SS.union premise_preds concl_preds in
    SS.iter (fun x -> print_endline @@ Predicate.show x) preds;
    let (arg1, _, _) = SS.choose preds
    and (arg2, _, _) = SS.choose preds in
    let res = Term.mkApp (Lazy.force impl1, [| Lazy.force entity_type; arg1; arg2 |]) in
    Format.printf "%a\n" pp_type res;
    Format.printf "ptr: %i\n" !ptr;
    *)
    (*
        let cexpr = Constrextern.extern_type true env sigma (Lazy.force imp) in
        let ident = Id.of_string "test" in
        let assum_kind = (Decl_kinds.Global, false, Decl_kinds.Logical) (* Axiom *) in
        let inline = Vernacexpr.NoInline in
        (* let is_coe = false *)
        (* and imps = [] (* implicit arguments *) *)
        (* let status = do_assumptions_bound_univs env loc_ident (Lazy.force imp) in *)
        (* if not status then Feedback.feedback Feedback.AddedAxiom; *)
        let pl = ((Loc.ghost, ident), None) in
        let l = [false, ([pl], cexpr)] in
        (* List.iter (fun (is_coe,(idl,c)) ->                    *)
        (* if Dumpglob.dump () then                              *)
        (* List.iter (fun (lid, _) ->                            *)
        (* Dumpglob.dump_definition lid false "ax") idl) l;      *)
        let _ = Command.do_assumptions assum_kind inline l in
    *)
     
    match axioms with
    | [] -> Tacticals.New.tclFAIL 0 (Pp.str"no new axiom generated.")
    | axioms ->
        Tacticals.New.tclTHEN
            (Proofview.Unsafe.tclEVARS !sigma)
            (List.fold_left (fun ts axm ->
                let name = new_axiom_name () in
                let axm = Tactics.assert_before (Name name) axm in
                (* Tacticals.New.tclTHEN *)
                    (Tacticals.New.tclTHENFIRST
                        (Tacticals.New.tclTHENFIRST ts axm)
                        (Tactics.Simple.case (Lazy.force devil)))
                (* (Tactics.red_in_hyp (name, Locus.InHyp)) *)
            ) Tacticals.New.tclIDTAC axioms)

    (* Command.declare_assumption is_coe assum_kind _ _ imps _ inline loc_ident *)
end }
    (* Tactics.assert_tac Anonymous res gl *)

let show_suggested () =
    let f _ cs =
        List.iter (fun c ->
        Printf.printf "CACHED: %s\n" (show_constr c)) cs
    in H.iter f cache


(*
let show_used_axiom c =
    (* Vernacentries.dump_global (AN c); *)
    let c = (Smartlocate.global_with_alias c) in
    let c = Universes.constr_of_global c in
    Printf.printf "%s\n" (show_constr c)

let show_used_axiom c =
    let open Misctypes in
    let c = AN c in
    Vernacentries.dump_global c;
    Feedback.msg_notice (Prettyp.print_name c)
    (* Printf.printf "%s\n" (show_constr c) *)
*)

let set_server addr check =
    server_address := addr;
    Printf.eprintf "Set Server Address: %s\n" !server_address;
    if check then begin
        let msg = default_echo ~msg:"Connecting" () in
        let encoder = Pbrt.Encoder.create () in
        encode_echo msg encoder;
        let res = Proc.communicate (Pbrt.Encoder.to_bytes encoder) !server_address in
        let { msg } = decode_echo (Pbrt.Decoder.of_bytes res) in
        if msg = "OK" then
            prerr_endline "Connection OK."
        else
            failwith (Printf.sprintf "unexpected message: %s\n" msg)
    end

let get_server () =
    Printf.eprintf "Server Address: %s\n" !server_address

open Stdarg
open Constrarg
open Extraargs

TACTIC EXTEND _ij_hello_
| [ "ij_hello" ] -> [ ij_hello ]
END

VERNAC COMMAND EXTEND AbductionServer CLASSIFIED AS QUERY
| [ "Abduction" "Set" "Server" string(s) ] -> [ set_server s true ]
| [ "Abduction" "Set" "Server" "NoCheck" string(s) ] -> [ set_server s false ]
| [ "Abduction" "Show" "Cache" ] -> [ show_suggested () ]
(* | [ "Abduction" "Show" "Used" global(c) ] ->
        [ show_used_axiom c ] *)
| [ "Abduction" "Set" "Debug" ] -> [ debug := true; prerr_endline "Abduction Debug True" ]
| [ "Abduction" "Reset" "Debug" ] -> [ debug := false; prerr_endline "Abduction Debug False" ]
| [ "Abduction" "Get" "Server" ] -> [ get_server () ]
END
