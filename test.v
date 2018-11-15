Require Export coqlib.
Require Export Abduction.Abduction.
Abduction Set Server "/tmp/py_server".
Parameter _hike : Event -> Prop.
Parameter _man : Entity -> Prop.
Parameter _woman : Entity -> Prop.
Parameter _walk : Event -> Prop.
Parameter _through : Event -> (Entity -> Prop).
Parameter _wooded : Entity -> Prop.
Parameter _area : Entity -> Prop.
Theorem t1: (and (exists x, (and (and (_man x) True) (exists e, (and (and (_hike e) ((Subj e) = x)) (exists x, (and (and (and (and (_wooded x) (_area x)) True) (_through e x)) True)))))) (exists x, (and (and (_woman x) True) (exists e, (and (and (_hike e) ((Subj e) = x)) (exists x, (and (and (and (and (_wooded x) (_area x)) True) (_through e x)) True))))))) -> (and (exists x, (and (and (_man x) True) (exists e, (and (and (_walk e) ((Subj e) = x)) (exists x, (and (and (and (and (_wooded x) (_area x)) True) (_through e x)) True)))))) (exists x, (and (and (_woman x) True) (exists e, (and (and (_walk e) ((Subj e) = x)) (exists x, (and (and (and (and (_wooded x) (_area x)) True) (_through e x)) True))))))).
nltac. Qed.
