
let p = Printf.printf

let pr = print_endline

let (!%) = Printf.sprintf

let read_lines file =
    let ch = open_in file in
    let rec parse () =
        let one = try Some (input_line ch)
                with End_of_file -> close_in_noerr ch; None
        in match one with
            | Some s -> s :: parse ()
            | None -> []
    in parse ()

let string_null s = String.length s = 0

let rec flatten = function
    [] -> []
    | head :: tail -> head @ flatten tail

let (--) a b =
    let rec iter store a bi =
        if a = bi then bi::store
        else iter (bi::store) a (bi - 1)
    in
    if a <= b then iter [] a b
    else List.rev @@ iter [] b a

let enumerate l = List.mapi (fun i elt -> (i, elt)) l

module Matrix =
struct
    type shape = int * int
    type 'a t = 'a array * shape

    let make (i, j) n = (Array.make (i*j) n, (i, j))

    (* let zeros s = make s 0.0 *)

    let of_list l = (Array.of_list l, (List.length l, 1))

    let length (_, (i, j)) = i * j

    let reshape (m, (i, j)) (i',j') =
        if i * j <> i' * j'
        then failwith (!%"reshape error: (%i, %i) to (%i, %i)" i j i' j')
        else (m, (i', j'))

    let get (m, (_, i)) (x, y) = m.(i * x + y)

    let set (m, (_, i)) (x, y) v = m.(i * x + y) <- v

    let init (i, j) f =
        let res = make (i, j) (f 0 0) in
        for i' = 0 to pred i do
            for j' = 0 to pred j do
                set res (i', j') (f i' j')
            done
        done;
        res

    let max_along_row (m, (_, i) as mat) x =
        let res = ref (get mat (x, 0)) in
        for idx = 1 to pred i do
            let v = get mat (x, idx) in
            match compare !res v with
            1 -> () | _ -> res := v
        done;
        !res
end


module type TypedMonadType =
sig
    type 'a t
    val empty : 'a t
    val singleton : 'a -> 'a t
    val bind : 'a t -> ('a -> 'b t) -> 'b t
end


module TypedMonad (S:TypedMonadType) =
struct
    open S
    let (>>=) = bind
    let (=<<) f m = m >>= f
    let fail = empty
    let return = singleton
    let rec sequence x = function
        | [] -> x
        | f::t -> sequence (x >>= f) t
    let sequence_ m l = sequence m l; ()
end

module Option =
struct
    module Core =
    struct
        type 'a t = 'a option
        let compare = compare
        let empty = None
        let singleton x = Some x
        let add x _ = Some x
        let bind x f = match x with
            | None -> None
            | Some x -> f x
        let to_list = function
            | None -> []
            | Some x -> [x]
        let fold f x e = match x with
            | None -> e
            | Some x -> f x e
    end

    include Core
    include TypedMonad(Core)
end
