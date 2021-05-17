import typing as ty
from dataclasses import dataclass

@dataclass
class CatalyticActivity:
    CSA: str
    UniprotIDs: str
    PDB: str
    EC: str
    residue: str
    aa: str
    chain: str
    resid: str
    function: str
    role: str
    roletype: str
    rolegroup: str

    @classmethod
    def from_line(cls, line: str):
        (CSA,
         UniprotID,
         PDB,
         EC,
         residue,
         aa,
         chain,
         resid,
         function,
         role,
         roletype,
         rolegroup) = line.split(",")
        return cls(CSA,
                   UniprotID,
                   PDB,
                   EC,
                   residue,
                   aa,
                   chain,
                   resid,
                   function,
                   role,
                   roletype,
                   rolegroup)


@dataclass
class ProteinResidueMapping:
    pdb_id: str
    chain_id: str
    residue_features: ty.Dict[int, CatalyticActivity]

    @classmethod
    def from_pdb_associated_csa(cls, list_of_csa: ty.List[CatalyticActivity]):
        residue_features: ty.Dict[int, CatalyticActivity] = {int(csa.resid): csa for csa in list_of_csa}
        return list_of_csa[0].PDBcode, list_of_csa[0].chain, residue_features


@dataclass
class Ecod:
    name: str
    domain_id: str
    manual_rep: str
    f_id: str
    pdb: str
    chain: str
    pdb_range: str
    seqid_range: str
    unp_acc: str
    arch_name: str
    x_name: str
    h_name: str
    t_name: str
    f_name: str
    asm_status: str
    ligand: str

    @classmethod
    def from_line(cls, line: str):
        uid, ecod_domain_id, manual_rep, f_id, pdb, chain, pdb_range, seqid_range, unp_acc, arch_name, x_name, h_name, t_name, f_name, asm_status, ligand = line.split(
            "\t")
        return cls(uid, ecod_domain_id, manual_rep, f_id, pdb, chain, pdb_range, seqid_range, unp_acc, arch_name,
                   x_name, h_name, t_name, f_name, asm_status, ligand)


@dataclass
class EcodInfo:
    ecod_classification: ty.Dict[str, Ecod]

    @classmethod
    def from_ecod_lines(cls, lines: ty.Iterable[str]):
        ecod_domains: ty.List[Ecod] = [Ecod.from_line(x) for x in lines]
        return cls(ecod_classification={x.name: x for x in ecod_domains})

    @classmethod
    def from_ecod_class_file(cls, filename: str = "ecod.latest.domains.txt"):
        return cls.from_ecod_lines(filter(lambda x: not x.startswith("#"), (open(filename, "r"))))


@dataclass
class ScopE:
    # cl=46456,cf=46457,sf=46458,fa=46459,dm=46460,sp=116748,px=113449
    name: str
    pdb_id: str
    chain_info: str
    scope_hier: str
    scop_id: int
    cl: int
    cf: int
    sf: int
    fa: int
    dm: int
    sp: int
    px: int

    @classmethod
    def from_line(cls, line: str):
        info = line.split("\t")
        info, class_info = info[:-1], [(lambda x: x.split("=")[-1])(x) for x in info[-1].split(",")]
        assert len(info) == 5
        assert len(class_info) == 7
        return cls(name=info[0],
                   pdb_id=info[1],
                   chain_info=info[2],
                   scope_hier=info[3],
                   scop_id=int(info[4]),
                   cl=int(class_info[0]),
                   cf=int(class_info[1]),
                   sf=int(class_info[2]),
                   fa=int(class_info[3]),
                   dm=int(class_info[4]),
                   sp=int(class_info[5]),
                   px=int(class_info[6]))


@dataclass
class ScopeInfo:
    scope_classification: ty.Dict[str, ScopE]

    @classmethod
    def from_scope_lines(cls, lines: ty.Iterable[str]):
        scope_domains: ty.List[ScopE] = [ScopE.from_line(x) for x in lines]
        return cls(scope_classification={x.name: x for x in scope_domains})

    @classmethod
    def from_scope_class_file(cls, filename: str = "dir.cla.scope.2.05-stable.txt"):
        return cls.from_scope_lines(filter(lambda x: not x.startswith("#"), (open(filename, "r"))))


@dataclass
class Domain:
    name: str
    c_class: int
    architecture: int
    topology: int
    super_family: int
    s35: int
    s60: int
    s95: int
    s100: int
    s100_count: int
    length: int
    resolution: float

    @classmethod
    def from_line(cls, line: str):
        info = line.split()
        assert len(info) == 12
        name = info[0]
        assert len(name) == 7
        return cls(name=name,
                   c_class=int(info[1]),
                   architecture=int(info[2]),
                   topology=int(info[3]),
                   super_family=int(info[4]),
                   s35=int(info[5]),
                   s60=int(info[6]),
                   s95=int(info[7]),
                   s100=int(info[8]),
                   s100_count=int(info[9]),
                   length=int(info[10]),
                   resolution=float(info[11]))


@dataclass
class DomainInfo:
    domains: ty.Dict[str, Domain]

    @classmethod
    def from_domainlist_lines(cls, lines: ty.Iterable[str]):
        domains: ty.List[Domain] = [Domain.from_line(x) for x in lines]
        return cls(domains={x.name: x for x in domains})

    @classmethod
    def from_domainlist_file(cls, filename: str):
        return cls.from_domainlist_lines(filter(lambda x: not x.startswith("#"), (open(filename, "r"))))
